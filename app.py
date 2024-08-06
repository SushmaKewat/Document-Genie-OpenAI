import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.llms import OpenAI

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory.buffer import ConversationBufferMemory

#from langchain.chains.question_answering.chain import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

import os
import shutil
from dotenv import load_dotenv
from itertools import groupby

load_dotenv()
# Dummy credentials
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("PASSWORD")

st.set_page_config(page_title="Doc Genie",page_icon="📑", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(text)
    ids = get_document_splits_with_ids(chunks)
    return chunks, ids

def get_document_splits_with_ids(doc_chunks):
    document_ids = []
    
    for page, chunks in groupby(doc_chunks, lambda chunk : chunk.metadata['page']):
        document_ids.extend([f"Source: {chunk.metadata['source'].split('/')[-1]}, Page no.: {page+1}, Chunk ID: {chunk_id}" for chunk_id, chunk in enumerate(chunks)])
    
    return document_ids

def get_vector_store(text_chunks, ids, api_key):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_documents(documents=text_chunks, ids=ids, embedding=embeddings)
        
        vector_store.save_local("faiss_index")
        
    except Exception as e:
        print(f'[Vector Store Error]: Could not create the vector store.')

def get_conversational_chain(api_key):
    prompt_template = """
    You are a firendly PDF assistant the helps the user to understand the contents of the uploaded files and give responses based on the content present in the context.
    For the first query, greet the user first.
    Answer the question/query correctly from the provided context, make sure to provide all the details. 
    If the user asks for the summary of the document, go through the contents of the document and give your response. 
    If you don't find the answer, go through the context again, even then if the answer is not in provided context just say, "Answer is not available in the context", don't provide the wrong answer and do not leave any answer unfinished. Always provide full answer and complete the sentence.
    \n\n
    =====BEGIN DOCUMENT=====
    {summaries}
    =====END DOCUMENT=====

    =====BEGIN CONVERSATION=====
    {conversation_memory}
    Question: \n{question}\n

    Answer:
    """

    history = StreamlitChatMessageHistory(key='chat_messages') # retrieve the history of the streamlit application
    memory = ConversationBufferMemory(chat_memory = history, input_key='question', memory_key='conversation_memory') # store the history in the memory
#Context:\n {context}\n
    # iterate over the history
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)

    model = ChatOpenAI(temperature=0.5, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_memory", "question", "summaries"])
    chain = load_qa_with_sources_chain(llm=model, chain_type="stuff", memory=memory, prompt=prompt)
  
    return chain

def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True) # Enable dangerous deserialization
    docs = db.similarity_search(query=user_question, fetch_k=5)
    
    chain = get_conversational_chain(api_key)

    # write the human and chatbot messages to the screen
    st.chat_message('human').markdown(f"**{user_question}**")
    
    response = chain({"input_documents": docs, "question": user_question})#, return_only_outputs=True)
    
    response_metadata = response["input_documents"][0].metadata
    metadata = [response_metadata['page']+1, response_metadata['source']]
    source = f'Page: {metadata[0]}, Source: {metadata[1]}'
    
    st.chat_message("ai").markdown(f"{response['output_text']}\n\n :gray[{source}]")
    

def login(username, password):
    return username == USERNAME and password == PASSWORD  

spacer_left, form, spacer_right = st.columns([1, 1, 1])

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.header("RAG Chatbot💁")
        st.markdown(""" ## Document Genie: Get instant insights from your Documents""")
        
        # This is the first API key input; no need to repeat it in the main function.
        api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
        
        st.markdown("""
                    ### How It Works

                    Follow these simple steps to interact with the chatbot:

                    1. **Enter Your API Key**: You'll need an OpenAI API key in the above input for the chatbot to access OpenAI models.

                    2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

                    3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.""")
        
        docs = []
        if not os.path.exists("docs"):
            os.mkdir("docs")
            
        with st.sidebar:
            st.title("Menu")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type=".pdf", accept_multiple_files=True, key="pdf_uploader")
            if st.button("Embed Documents", key="process_button") and api_key:  # Check if API key is provided before processing
                with st.spinner("Processing..."):
                    for pdf in pdf_docs:
                        #raw_text = get_pdf_text(pdf_docs)
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
            
            if st.button("Reset Chat"):
                st.session_state.chat_messages = []
                st.rerun()
                
            if st.button("Logout", key="logout_button"):
                shutil.rmtree("faiss_index", ignore_errors=True)
                shutil.rmtree("docs", ignore_errors=True)
                st.session_state.logged_in = False
                st.rerun()

        if x:=st.chat_input("Ask a Question from the PDF Files", key='user_question'): # input for continuous conversation
       
            if x and api_key:  # Ensure API key and x, that is,user question are provided
                 user_input(x, api_key) # this function is called every time a user enters a query

            

    else:
        with form:
            with st.container(border=True):
                st.title("Login Page")
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
