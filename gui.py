import streamlit as st
from app import retrieve_answers

def header():
    header = st.container()
    header.title("Q&A GenAI BOT")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    ### Custom CSS for the sticky header
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            z-index: 999;
            background: #FFFFFF;
        }
        .fixed-header {
            border-bottom: 1.3px solid;
        }
    </style>
        """,
        unsafe_allow_html=True
    )

def run_app():

    st.set_page_config(page_title="GenAI Q&A BOT")
    # st.title("Q&A GenAI BOT")
    header()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter you Question"):

        # Display user message in chat message container
        st.chat_message("user").markdown('**You :** <br/><br/>' + prompt, unsafe_allow_html=True)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response_unformatted = retrieve_answers(st.session_state.messages, prompt)
        response_formatted = response_unformatted["output_text"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown('**Q&A BOT :** <br/><br/>'+ response_formatted, unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_formatted})