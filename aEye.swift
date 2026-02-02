import UIKit
import AVFoundation
import Vision
import CoreML

final class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    // --- CONFIG (matching your Python constants) ---
    let confThresh: Float = 0.60
    let distCalib: CGFloat = 1500
    let speechDelay: TimeInterval = 8

    // --- Camera ---
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let visionQueue = DispatchQueue(label: "vision.queue")

    // --- Vision / ML ---
    private var objectRequest: VNCoreMLRequest!
    private let poseRequest = VNDetectHumanBodyPoseRequest()

    // --- Speech ---
    private let speaker = AVSpeechSynthesizer()
    private var lastSummaryTime: Date = .distantPast

    // --- “Learned items” (Vision feature prints) ---
    // This replaces ORB matching in a way that fits iOS.
    private var learnedPrints: [String: VNFeaturePrintObservation] = [:]
    private let learnedThreshold: Float = 15.0 // smaller = stricter; tune this

    // --- Overlay ---
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let overlayLayer = CALayer()
    
    private var latestPixelBuffer: CVPixelBuffer?

    override func viewDidLoad() {
        super.viewDidLoad()

        setupPreview()
        loadLearnedImagesFromBundle()
        setupObjectDetectionModel()
        setupCamera()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayLayer.frame = view.bounds
    }

    private func setupPreview() {
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)

        overlayLayer.frame = view.bounds
        view.layer.addSublayer(overlayLayer)
    }

    // Load learned images (e.g., "mug.jpg") from app bundle.
    private func loadLearnedImagesFromBundle() {
        // Put images in your Xcode target’s bundle.
        // Example: mug.jpg, keys.jpg, etc.
        let learnedNames: [String] = [] // <-- rename to your assets

        for name in learnedNames {
            guard let url = Bundle.main.url(forResource: name, withExtension: "jpg"),
                  let uiImage = UIImage(contentsOfFile: url.path),
                  let cgImage = uiImage.cgImage else { continue }

            if let fp = makeFeaturePrint(from: cgImage) {
                learnedPrints[name] = fp
            }
        }
    }

    private func makeFeaturePrint(from cgImage: CGImage) -> VNFeaturePrintObservation? {
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
            return request.results?.first as? VNFeaturePrintObservation
        } catch {
            print("Feature print error:", error)
            return nil
        }
    }

    private func setupObjectDetectionModel() {
        do {
            let config = MLModelConfiguration()

            // ✅ Replace "MyModelClassName" with the CoreML class name Xcode shows
            let mlModel = try yolo26n(configuration: config).model

            let vnModel = try VNCoreMLModel(for: mlModel)

            objectRequest = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
                self?.handleObjectDetections(request: request, error: error)
                print("Object request results:", request.results?.map { String(describing: type(of: $0)) } ?? [])
                print("Object results count:", request.results?.count ?? 0)

            }

            objectRequest.imageCropAndScaleOption = .scaleFill
            print("✅ Object detection model loaded")
        } catch {
            print("❌ Failed to load model:", error)
        }
    }

    private func setupCamera() {
        sessionQueue.async {
            self.session.beginConfiguration()
            self.session.sessionPreset = .high

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                print("❌ Camera input failed")
                return
            }
            self.session.addInput(input)

            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.setSampleBufferDelegate(self, queue: self.visionQueue)

            guard self.session.canAddOutput(self.videoOutput) else {
                print("❌ Output add failed")
                return
            }
            self.session.addOutput(self.videoOutput)

            if let conn = self.videoOutput.connection(with: .video) {
                conn.videoRotationAngle = 90 // portrait
            }

            self.session.commitConfiguration()
            self.session.startRunning()
        }
    }
    
    private func exifOrientationForCurrentDevice() -> CGImagePropertyOrientation {
        // Back camera in portrait is usually .right
        // (This is the #1 reason pose returns 0)
        return .right
    }


    // MARK: - Frame processing

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestPixelBuffer = pixelBuffer
        guard objectRequest != nil else { return }
        
        let exif = exifOrientationForCurrentDevice()

        // 1) Run object detection (your custom YOLO CoreML)
        let objectHandler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: exif,
            options: [:]
        )

        // 2) Run pose detection (Apple Vision)
        
        let poseHandler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: exif,
            options: [:]
        )

        do {
            try objectHandler.perform([objectRequest])
            try poseHandler.perform([poseRequest])
        } catch {
            print("Vision error:", error)
            return
        }

        // 3) Learned item matching (feature print)
        let learnedAlert = findLearnedItem(pixelBuffer: pixelBuffer)

        // 4) Build summary + speak occasionally (we fill these from handlers)
        // We store latest detections in properties updated from handlers.
        // For simplicity, this sample calls speak from main after drawing.
        DispatchQueue.main.async {
            self.drawOverlaysAndMaybeSpeak(learnedAlert: learnedAlert)
        }
    }

    // MARK: - Object detection results storage

    private struct DetectedThing {
        var label: String
        var distanceM: Int
        var midX: CGFloat
        var box: CGRect
    }

    private struct DetectedPerson {
        var action: String
        var distanceM: Int
        var midX: CGFloat
        var box: CGRect
        var joints: [VNHumanBodyPoseObservation.JointName: CGPoint]
    }

    private var furnitureFound: [DetectedThing] = []
    private var otherItems: [DetectedThing] = []
    private var tableItems: [String] = []
    private var peopleFound: [DetectedPerson] = []
    private var tableBox: CGRect?

    private func handleObjectDetections(request: VNRequest, error: Error?) {
        furnitureFound.removeAll()
        otherItems.removeAll()
        tableItems.removeAll()
        tableBox = nil
        // ⚠️ do not clear peopleFound here (pose fills it)

        guard let feats = request.results as? [VNCoreMLFeatureValueObservation],
              let arr = feats.first?.featureValue.multiArrayValue else { return }

        // Expect shape [1, 300, 6]
        let d0 = arr.shape.count > 0 ? arr.shape[0].intValue : 0
        let d1 = arr.shape.count > 1 ? arr.shape[1].intValue : 0
        let d2 = arr.shape.count > 2 ? arr.shape[2].intValue : 0
        if !(d0 == 1 && d1 == 300 && d2 == 6) {
            print("Unexpected YOLO shape:", arr.shape)
            return
        }

        // Determine if coords are normalized or pixel-based
        let x2Sample = arr[[0, 0, 2] as [NSNumber]].doubleValue
        let normalizedCoords = x2Sample <= 1.5

        // Labels: your model class names
        // If your CoreML model class has a `labels` property, use it instead.
        // For Ultralytics exports, you may not have names here; you can map manually.
        let names = (try? yolo26n(configuration: MLModelConfiguration()).model.modelDescription.classLabels as? [String]) ?? []

        // Convert detections
        for i in 0..<300 {
            let x1 = arr[[0, i, 0] as [NSNumber]].doubleValue
            let y1 = arr[[0, i, 1] as [NSNumber]].doubleValue
            let x2 = arr[[0, i, 2] as [NSNumber]].doubleValue
            let y2 = arr[[0, i, 3] as [NSNumber]].doubleValue
            let conf = arr[[0, i, 4] as [NSNumber]].doubleValue
            let cls  = Int(arr[[0, i, 5] as [NSNumber]].doubleValue)

            if conf < Double(confThresh) { continue }
            if x2 <= x1 || y2 <= y1 { continue }

            // Build a normalized metadata rect (0..1, origin top-left) for preview conversion
            let nx1: CGFloat
            let ny1: CGFloat
            let nw: CGFloat
            let nh: CGFloat

            if normalizedCoords {
                nx1 = CGFloat(x1)
                ny1 = CGFloat(y1)
                nw  = CGFloat(x2 - x1)
                nh  = CGFloat(y2 - y1)
            } else {
                // If pixel coords, convert using current pixelBuffer size
                guard let pb = latestPixelBuffer else { continue }
                let w = CGFloat(CVPixelBufferGetWidth(pb))
                let h = CGFloat(CVPixelBufferGetHeight(pb))
                nx1 = CGFloat(x1) / w
                ny1 = CGFloat(y1) / h
                nw  = CGFloat(x2 - x1) / w
                nh  = CGFloat(y2 - y1) / h
            }

            let bboxNorm = CGRect(x: nx1, y: ny1, width: nw, height: nh)
            let rect = previewLayer.layerRectConverted(fromMetadataOutputRect: bboxNorm)

            let label = (cls >= 0 && cls < names.count) ? names[cls] : "cls\(cls)"

            // You can split furniture vs items like before; for now just store as otherItems
            let dist = Int((distCalib / max(rect.width, 1)).rounded())
            otherItems.append(.init(label: label, distanceM: dist, midX: rect.midX, box: rect))
        }
    }

    private func findLearnedItem(pixelBuffer: CVPixelBuffer) -> String? {
        // Generate a feature print for the current frame and compare to learned prints.
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: exifOrientationForCurrentDevice(),
                                            options: [:])

        do {
            try handler.perform([request])
            guard let frameFP = request.results?.first as? VNFeaturePrintObservation else { return nil }

            var bestName: String?
            var bestDistance: Float = .greatestFiniteMagnitude

            for (name, learnedFP) in learnedPrints {
                var d: Float = 0
                try frameFP.computeDistance(&d, to: learnedFP)
                if d < bestDistance {
                    bestDistance = d
                    bestName = name
                }
            }

            if let bestName, bestDistance < learnedThreshold {
                return bestName
            }
            return nil
        } catch {
            return nil
        }
    }

    // MARK: - Overlay + Pose + Speech

    private func drawOverlaysAndMaybeSpeak(learnedAlert: String?) {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        let poseObs = poseRequest.results ?? []
        print("Pose observations:", poseObs.count)


        // --- Draw furniture + items ---
        for f in furnitureFound {
            drawBox(f.box, text: "\(f.label) \(f.distanceM)m", color: UIColor.blue.cgColor)
        }
        for it in otherItems {
            drawBox(it.box, text: "\(it.label) \(it.distanceM)m", color: UIColor.yellow.cgColor)
        }

        // --- Pose detection and actions ---
        let poseResults = poseRequest.results ?? []
            for pose in poseResults {
                guard let pb = latestPixelBuffer else { continue }
                let joints = extractJoints(pose, pixelBuffer: pb)
                let action = inferActionFromPose(joints: joints)
                // We don’t have person boxes from Vision Pose by default; keep it simple:
                drawSkeleton(joints: joints)
                // You *can* also combine with “person” boxes from your object model if your model outputs them.
                peopleFound.append(.init(action: action, distanceM: 0, midX: 0, box: .zero, joints: joints))
            }

        // --- Speak summary every speechDelay seconds ---
        let now = Date()
        if now.timeIntervalSince(lastSummaryTime) > speechDelay {
            let text = buildSpeechSummary(learnedAlert: learnedAlert)
            if !text.isEmpty {
                speak(text)
                lastSummaryTime = now
            }
        }

        // --- Top-left summary overlay text ---
        drawSummaryText(learnedAlert: learnedAlert)
    }

    private func extractJoints(_ obs: VNHumanBodyPoseObservation, pixelBuffer: CVPixelBuffer) -> [VNHumanBodyPoseObservation.JointName: CGPoint] {
        var out: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
        guard let all = try? obs.recognizedPoints(.all) else { return out }

        let minConf: Float = 0.15
        let exif = exifOrientationForCurrentDevice()

        for (name, p) in all {
            guard p.confidence >= minConf else { continue }

            var x = CGFloat(p.location.x)
            var y = CGFloat(p.location.y)

            // Vision joints are in normalized coords with origin bottom-left.
            // Convert to metadata-style normalized coords with origin top-left,
            // AND apply rotation based on EXIF.
            switch exif {
            case .right:
                // rotate 90 CW
                let xr = y
                let yr = 1.0 - x
                x = xr
                y = yr
            case .left:
                // rotate 90 CCW
                let xl = 1.0 - y
                let yl = x
                x = xl
                y = yl
            case .up:
                y = 1.0 - y
            case .down:
                let xd = 1.0 - x
                let yd = y
                x = xd
                y = yd
                y = 1.0 - y
            default:
                // mirrored cases can be added later if you switch to front camera
                y = 1.0 - y
            }

            let normRect = CGRect(x: x, y: y, width: 0.001, height: 0.001)
            let converted = previewLayer.layerRectConverted(fromMetadataOutputRect: normRect)
            out[name] = CGPoint(x: converted.midX, y: converted.midY)
        }

        return out
    }

    private func inferActionFromPose(joints: [VNHumanBodyPoseObservation.JointName: CGPoint]) -> String {
        var actions = Set<String>()

        let waveMargin: CGFloat = 40   // increase if it still false-triggers
        let sitMargin: CGFloat = 50
        let lieMargin: CGFloat = 90

        // --- Waving: wrist clearly above shoulder ---
        if let lw = joints[.leftWrist], let ls = joints[.leftShoulder],
           lw.y < (ls.y - waveMargin) {
            actions.insert("waving")
        }
        if let rw = joints[.rightWrist], let rs = joints[.rightShoulder],
           rw.y < (rs.y - waveMargin) {
            actions.insert("waving")
        }

        // --- Sitting: knee close to hip vertically (rough) ---
        if let lh = joints[.leftHip], let lk = joints[.leftKnee],
           abs(lh.y - lk.y) < sitMargin {
            actions.insert("sitting")
        }
        if let rh = joints[.rightHip], let rk = joints[.rightKnee],
           abs(rh.y - rk.y) < sitMargin {
            actions.insert("sitting")
        }

        // --- Lying down: shoulders and hips close in vertical direction ---
        if let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
           let lh = joints[.leftHip], let rh = joints[.rightHip] {

            let shoulderY = (ls.y + rs.y) / 2
            let hipY = (lh.y + rh.y) / 2
            if abs(shoulderY - hipY) < lieMargin {
                actions.insert("lying down")
            }
        }

        if actions.isEmpty { return "unknown" }
        return actions.sorted().joined(separator: " and ")
    }
    
    private func poseBoundingBox(joints: [VNHumanBodyPoseObservation.JointName: CGPoint]) -> CGRect? {
        let pts = Array(joints.values)
        guard pts.count >= 4 else { return nil } // need enough points

        let xs = pts.map { $0.x }
        let ys = pts.map { $0.y }

        let minX = xs.min()!, maxX = xs.max()!
        let minY = ys.min()!, maxY = ys.max()!

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }


    private func buildSpeechSummary(learnedAlert: String?) -> String {
        var parts: [String] = []

        if let learnedAlert { parts.append("I found your \(learnedAlert)!") }

        if let f = furnitureFound.first {
            parts.append("a \(f.label) \(f.distanceM) meters away")
        }

        if let p = peopleFound.first {
            if p.action == "unknown" {
                parts.append("a person")
            } else {
                parts.append("a person who is \(p.action)")
            }
        }

        if let firstOnTable = tableItems.first {
            parts.append("a table with a \(firstOnTable) on it")
        }

        if let it = otherItems.first {
            parts.append("a \(it.label) \(it.distanceM) meters away")
        }

        if parts.isEmpty { return "" }
        return "I see " + parts.joined(separator: ", and ") + "."
    }

    private func speak(_ text: String) {
        guard !speaker.isSpeaking else { return }
        let utt = AVSpeechUtterance(string: text)
        utt.rate = 0.5
        speaker.speak(utt)
    }

    // MARK: - Drawing helpers

    private func drawBox(_ rect: CGRect, text: String, color: CGColor) {
        let box = CAShapeLayer()
        box.frame = rect
        box.path = UIBezierPath(rect: box.bounds).cgPath
        box.strokeColor = color
        box.fillColor = UIColor.clear.cgColor
        box.lineWidth = 2
        overlayLayer.addSublayer(box)

        let label = CATextLayer()
        label.string = text
        label.fontSize = 14
        label.foregroundColor = color
        label.backgroundColor = UIColor.black.withAlphaComponent(0.6).cgColor
        label.frame = CGRect(x: rect.minX, y: rect.minY - 22, width: min(260, rect.width + 120), height: 20)
        label.contentsScale = UIScreen.main.scale
        overlayLayer.addSublayer(label)
    }

    private func drawSkeleton(joints: [VNHumanBodyPoseObservation.JointName: CGPoint]) {
        func line(_ a: VNHumanBodyPoseObservation.JointName, _ b: VNHumanBodyPoseObservation.JointName) {
            guard let p1 = joints[a], let p2 = joints[b] else { return }
            let path = UIBezierPath()
            path.move(to: p1)
            path.addLine(to: p2)

            let layer = CAShapeLayer()
            layer.path = path.cgPath
            layer.strokeColor = UIColor.green.cgColor
            layer.lineWidth = 2
            overlayLayer.addSublayer(layer)
        }

        // Similar to your connection list
        line(.leftShoulder, .rightShoulder)
        line(.leftShoulder, .leftElbow)
        line(.leftElbow, .leftWrist)
        line(.rightShoulder, .rightElbow)
        line(.rightElbow, .rightWrist)
        line(.leftHip, .rightHip)
        line(.leftHip, .leftKnee)
        line(.leftKnee, .leftAnkle)
        line(.rightHip, .rightKnee)
        line(.rightKnee, .rightAnkle)
        line(.leftShoulder, .leftHip)
        line(.rightShoulder, .rightHip)

        // Draw joint dots
        for (_, p) in joints {
            let dot = CAShapeLayer()
            dot.path = UIBezierPath(ovalIn: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)).cgPath
            dot.fillColor = UIColor.red.cgColor
            overlayLayer.addSublayer(dot)
        }
    }

    private func drawSummaryText(learnedAlert: String?) {
        let summary = CATextLayer()
        summary.contentsScale = UIScreen.main.scale
        summary.frame = CGRect(x: 12, y: 50, width: view.bounds.width - 24, height: 120)
        summary.backgroundColor = UIColor.black.withAlphaComponent(0.6).cgColor
        summary.foregroundColor = UIColor.white.cgColor
        summary.fontSize = 14

        var lines: [String] = ["DETECTIONS:"]
        if let learnedAlert { lines.append("• Learned Item: \(learnedAlert)") }
        for f in furnitureFound.prefix(3) { lines.append("• \(f.label) (\(f.distanceM)m)") }
        if let p = peopleFound.first {
            if p.action == "unknown" {
                lines.append("• Person")
            } else {
                lines.append("• Person \(p.action)")
            }
        }
        for it in otherItems.prefix(3) { lines.append("• \(it.label) (\(it.distanceM)m)") }

        summary.string = lines.joined(separator: "\n")
        overlayLayer.addSublayer(summary)
    }
}
