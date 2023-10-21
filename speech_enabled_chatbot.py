#import necessary libraries
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

import speech_recognition as sr
import streamlit as st

import nltk
from nltk.stem import WordNetLemmatizer

# for downloading packages
nltk.download('popular', quiet=True)

# uncomment the following only the first time
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

def transcribe_speech():
    # Initialize recognizer class
    # Creating a Recognizer instance
    r = sr.Recognizer()
    
    # Creating an instance of the Microphone class
    mic = sr.Microphone()
    
    # Reading Microphone as source
    with mic as source:
        # Handle ambient noise
        r.adjust_for_ambient_noise(source, duration=0.5)
        
        st.info("Speak now...")
        
        # listen for speech and store in audio_text variable
        recorded_audio = r.listen(source, timeout=None)  # Use timeout=None to listen indefinitely
        
        st.info("Transcribing...")

        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio_text)`
            text = r.recognize_google(recorded_audio)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"


#Reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as f:
    raw = f.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences 
word_tokens = nltk.word_tokenize(raw) # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

def main():
    st.set_page_config(layout="wide")
    
    st.title("Speech Enabled Chatbot")
    st.write("CLOUD: My name is Cloud. I will answer your queries about Chatbots. If you want to exit, type or say Bye!")
    st.write("By the way, you can choose the way you ask me : Text or using your own voice!")
    # Create a sidebar
    st.sidebar.title("How do you want to communicate with CLOUD, our lovely chatbot?")
    
    # Choose input type
    input_type = st.sidebar.selectbox("Choose Input [Text/Speech]", ["Text", "Speech"])
    
    if input_type == "Text":
        user_response = st.text_input("You: ")
        user_response = user_response.lower()

        if st.button("submit"):
            if(user_response!='bye'):
                if(user_response=='thanks' or user_response=='thank you'):
                    flag=False
                    st.write("CLOUD: You are welcome..")
                else:
                    if(greeting(user_response)!=None):
                        st.write("CLOUD: " + greeting(user_response))
                    else:
                        st.write("CLOUD: " + response(user_response))
                        sent_tokens.remove(user_response)
            else:
                st.write("CLOUD: Bye! take care..")
                
    elif input_type == "Speech":
        # add a button to trigger speech recognition
        if st.button("Start Recording"):
            user_response = transcribe_speech()
            user_response = user_response.lower()

            if(user_response!='bye'):
                if(user_response=='thanks' or user_response=='thank you'):
                    flag=False
                    st.write("CLOUD: You are welcome..")
                else:
                    if(greeting(user_response)!=None):
                        st.write("CLOUD: " + greeting(user_response))
                    else:
                        st.write("CLOUD: " + response(user_response))
                        sent_tokens.remove(user_response)
            else:
                st.write("CLOUD: Bye! take care..")
            
if __name__=='__main__':
    main()