import nltk
import streamlit as st
import time
import speech_recognition as sr
from chatbot2 import preprocess, chatbot_response


# load the text file
with open('human_text.txt','r', encoding='utf8', errors='ignore') as f:
    data = f.read()

# preprocess the data
processed_data = preprocess(data)

def transcribe_speech():
    # initialize recognizer class
    r = sr.Recognizer()

    # reading microphone as source
    with sr.Microphone() as source:
        # create a spinner that shows progress
        with st.spinner(text='Silence please, Calibrating background noise.....'):
            time.sleep(3)

        r.adjust_for_ambient_noise(source, duration=1)
        st.info("Speak now...")

        audio_text = r.listen(source)  # listen for speech and store in audio_text variable

        with st.spinner(text='Transcribing your voice to text'):
            time.sleep(2)

        try:
            # using Google speech recognition to recognize the audio
            text = r.recognize_google(audio_text)
            return text
        except:
            return "Sorry, I did not understand what you said."

def chatbot(input_text):
    if input_text.lower() == 'exit':
        return "Goodbye!"

    response = chatbot_response(input_text, processed_data)
    return response

st.title("Speech-enabled Chatbot")

# add a radio button to select the input mode
input_mode = st.radio("Select input mode:", ("Text", "Speech"))

if input_mode == "Text":
    # add a text input box for text input
    input_text = st.text_input("You: ")

    if st.button("Send"):
        # get the chatbot response
        response = chatbot(input_text)
        st.text_area("SCBot:", value=response, height=100)
else:
    # add a button to start speech recognition
    if st.button("Start recording"):
        # transcribe the speech into text
        input_text = transcribe_speech()

        # get the chatbot response
        response = chatbot(' '.join(preprocess(input_text)))
        st.text_area("SCBot:", value=response, height=100)

