import streamlit as st
from app import retrieve_answers

def run_app():
    
    # Set page config and theme
    st.set_page_config(page_title="GenAI Q&A BOT", layout="wide", initial_sidebar_state="collapsed")

    #Set Page Title   
    st.title("Q&A GenAI BOT")

    # Create a text input for the question
    question = st.text_input("Enter your question:")
    resp = retrieve_answers(question)

    # Create a Submit button
    submit = st.button("Ask the Question")

    # If submit button is clicked
    if submit:
        st.subheader("The Response is")
        st.write(resp["output_text"])