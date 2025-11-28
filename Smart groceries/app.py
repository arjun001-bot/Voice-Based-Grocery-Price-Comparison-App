import streamlit as st
from voice_listen import listen_once

st.title("Week 1: Voice-based Shopping List Demo")
st.markdown("Speak your grocery items and see the list appear below.")

if st.button("Use Voice (local)"):
    st.info("Listening... speak now.")
    spoken = listen_once()
    if spoken.startswith("[STT error]"):
        st.error(spoken)
    elif spoken.strip() == "":
        st.warning("No speech recognized.")
    else:
        # Split by common separators
        items_text = spoken.replace(" and ", "\n").replace(",", "\n")
        st.success("Your spoken items:")
        for item in items_text.splitlines():
            st.write("-", item)
