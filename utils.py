import os
import shutil
import streamlit as st
from itertools import groupby
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_core.prompts.prompt import PromptTemplate

load_dotenv()

def remove_files():
    shutil.rmtree("faiss_index", ignore_errors=True)
    shutil.rmtree("docs", ignore_errors=True)
    
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
    For the first query, greet the user.
    Answer the question/query correctly from the provided context, make sure to provide all the details. 
    If the user asks for the summary of the document, go through the contents of the document and give your response. 
    If you don't find the answer, go through the context again, even then if the answer is not in provided context just say, "Answer is not available in the context", don't provide the wrong answer and do not leave any answer unfinished. Always provide full answer and complete the sentence.
    Try to answer in bullets wherever neccessary.
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

    # iterate over the history
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)

    model = ChatOpenAI(temperature=0.5, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_memory", "question", "summaries"])
    chain = load_qa_with_sources_chain(llm=model, chain_type="stuff", memory=memory, prompt=prompt)
  
    return chain

def login(username, password):
    # credentials
    USERNAME = os.getenv("APP_USERNAME")
    PASSWORD = os.getenv("PASSWORD")
    
    return username == USERNAME and password == PASSWORD 