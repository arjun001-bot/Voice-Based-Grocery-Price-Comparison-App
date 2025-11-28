import speech_recognition as sr

def listen_once(timeout=6, phrase_time_limit=8):
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        print("Listening... speak your shopping list:")
        audio = r.listen(mic, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = r.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"[STT error] {e}"
