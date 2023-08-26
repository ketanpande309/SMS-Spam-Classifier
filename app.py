import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1.Preprocess
    transformed_text = transform_text(input_sms)

    # 2.Vectorize
    vectorized_text = tfidf.transform([transformed_text])

    # 3.Predict
    result = model.predict(vectorized_text)[0]

    # 4.Display
    if result == 1:
        st.header('Spam SMS')

    else:
        st.header('Not Spam SMS')
