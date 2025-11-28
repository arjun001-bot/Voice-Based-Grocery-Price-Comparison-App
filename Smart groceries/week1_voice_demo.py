import streamlit as st
import speech_recognition as sr

st.title("Week 1: Voice-based Shopping List Demo")
st.write("Click the button and speak the product names. They will appear below.")

def listen_once():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        st.info("Listening... speak your shopping list now!")
        audio = r.listen(mic, timeout=6, phrase_time_limit=8)
    try:
        text = r.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"[STT error] {e}"

if st.button("Use Voice (local)"):
    spoken_text = listen_once()
    if spoken_text.startswith("[STT error]"):
        st.error(spoken_text)
    elif spoken_text.strip() == "":
        st.warning("No speech recognized.")
    else:
        # Split spoken items by "and" or comma
        items = [item.strip() for item in spoken_text.replace(" and ", ",").split(",") if item.strip()]
        st.success("Recognized items:")
        for i, item in enumerate(items, 1):
            st.write(f"{i}. {item}")
