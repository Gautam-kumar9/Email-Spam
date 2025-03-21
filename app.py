import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # Ensure punkt is downloaded

    y = []
    for i in text:
        if i.isalnum():  # Remove punctuations
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))  # Fetch stopwords only once

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Ensure 'vectorizer.pkl' and 'model.pkl' exist.")

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_email = st.text_area("Enter the message: ")

if st.button('Predict'):
    if input_email.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_mail = transform_text(input_email)
        vector_input = tfidf.transform([transformed_mail])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")
