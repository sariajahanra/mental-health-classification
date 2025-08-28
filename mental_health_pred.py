import streamlit as st
import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
svm_model = joblib.load("svm_depression_model.pkl")

st.set_page_config(layout="wide") # This makes the app fill the whole browser width

st.title("Mental Health Tweet Analyzer")

# Use a container to visually separate the input section
with st.container(border=True):
    col1, col2 = st.columns([3, 1]) # Adjust column ratios as needed

    with col1:
        st.subheader("Enter a Tweet for Analysis")
        tweet = st.text_area("Tweet:", placeholder="Type your tweet here...", label_visibility="hidden")
        
    with col2:
        st.subheader("Action")
        st.markdown("<br>", unsafe_allow_html=True) # Add some space
        if st.button("Predict Sentiment", type="primary"):
            if tweet.strip():
                tweet_tfidf = vectorizer.transform([tweet])
                prediction = svm_model.predict(tweet_tfidf)[0]
                
                label = "Depressed 😞" if prediction == 1 else "Non-Depressed 🙂"
                st.success(f"**Prediction:** {label}")
            else:
                st.warning("Please enter some text.")