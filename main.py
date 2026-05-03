import streamlit as st
import string
import pickle
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

def filtered_text(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for char in message:
        if char.isalnum():
            y.append(char)

    message = y[:]
    y.clear()

    for word in message:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    message = y[:]
    y.clear()

    for word in message:
        y.append(ps.stem(word))

    return " ".join(y)


tfidf = pickle.load(open("vectorized.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


st.title("Spam Email/Message Detector")

input_message = st.text_area("Enter Email/SMS")

if st.button("Detect"):

    if input_message.strip() == "":
        st.warning("Please Enter a Message")
    else:
        filtered_message = filtered_text(input_message)
        vectorized_message = tfidf.transform([filtered_message])

        result = model.predict(vectorized_message)[0]

        if result == 1:
            st.header("It's Spam")
        else:
            st.header("It's not Spam")


st.markdown("---", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:25px; font-family:Georgia, serif; font-size:18px; color:black; line-height:1.8;">
    Built by  <b>Mohsin</b><br>
    To connect with me on LinkedIn,
    <a href="https://www.linkedin.com/in/mohsinsp/" target="_blank" 
       style="color:blue; text-decoration:underline; font-weight:bold;">
       Click Here
    </a>
</div>
""", unsafe_allow_html=True)