import streamlit as st
import os

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from utils import remove_files, login, get_conversational_chain, get_vector_store, get_text_chunks

st.set_page_config(page_title="Doc Genie",page_icon="📑", layout="wide")   
 

def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True) # Enable dangerous deserialization
    docs = db.similarity_search(query=user_question, fetch_k=7)
    
    chain = get_conversational_chain(api_key)

    # write the human and chatbot messages to the screen
    st.chat_message('human').markdown(f"**{user_question}**")
    
    response = chain({"input_documents": docs, "question": user_question})#, return_only_outputs=True)
    
    response_metadata = response["input_documents"][0].metadata
    metadata = [response_metadata['page']+1, response_metadata['source']]
    source = f'Page: {metadata[0]}, Source: {metadata[1]}'
    
    st.chat_message("ai").markdown(f"{response['output_text']}\n\n :gray[{source}]")

spacer_left, form, spacer_right = st.columns([0.5, 2, 0.5])

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.markdown(""" ## Document Genie 💁: Get instant insights from your Documents""")
        
        # This is the first API key input; no need to repeat it in the main function.
        api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
        
        st.markdown("""
                    ### How It Works

                    Follow these simple steps to interact with the chatbot:

                    1. **Enter Your API Key**: You'll need an OpenAI API key in the above input for the chatbot to access OpenAI models.

                    2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

                    3. **Ask a Question**: After processing the documents, you can ask any question related to the content of your uploaded documents.""")
        
        docs = []
        if not os.path.exists("docs"):
            os.mkdir("docs")
            
        with st.sidebar:
            st.title("Menu")
            pdf_docs = st.file_uploader("Upload your PDF files and click the Submit button", type=".pdf", accept_multiple_files=True, key="pdf_uploader")
            if st.button("Submit", use_container_width=True, key="process_button") and api_key:  # Check if API key is provided before processing
                with st.spinner("Processing..."):
                    for pdf in pdf_docs:
                        filepath = os.path.join("docs/", pdf.name)
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        with open(filepath, "wb") as f:
                            f.write(pdf.getbuffer())
                            
                        loader = PyPDFLoader(filepath)
                        docs.extend(loader.load())
                        text_chunks, ids = get_text_chunks(docs)
                        get_vector_store(text_chunks, ids, api_key)
                        
                        st.success(f"{pdf.name} embedded successfully.")
                    st.caption("You can now ask questions from the uploaded documents.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
            
            with col2:       
                if st.button("Logout", use_container_width=True, key="logout_button"):
                    remove_files()
                    st.session_state.logged_in = False
                    st.rerun()
    
            
                
        if x:=st.chat_input("Ask a Question from the PDF Files", key='user_question'): # input for continuous conversation
       
            if x and api_key:  # Ensure API key and x, that is,user question are provided
                 user_input(x, api_key) # this function is called every time a user enters a query 

    else:
        with form:
            with st.container(border=True):
                st.title("Login")
                st.caption("Please enter the credentials to continue.")

                # Create login form
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                if st.button("Login"):
                    if login(username, password):
                        st.session_state.logged_in = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

if __name__ == "__main__":
    main()
