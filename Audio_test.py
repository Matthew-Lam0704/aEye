import pyttsx3 as pyt
engine = pyt.init(driverName='nsss')
engine.say("Hello, I am working on your Mac now.")
engine.runAndWait()