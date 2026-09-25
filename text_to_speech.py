import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()
    

speak("Hola, estoy usando python para convertir texto a voz")