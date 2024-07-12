"""Python file to serve as the frontend"""
import sys
import os
sys.path.append(os.path.abspath('.'))

import time

from langchain.chains import ConversationChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pathlib import Path
from PyPDF2 import PdfReader
import datetime
from urllib.parse import quote

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from pydantic.v1 import BaseModel, Field
import pandas as pd
import datetime
import time
import pdfplumber
import json
import io
import openai

openai.api_type = "azure"
openai.api_base = "https://pn-llm-1.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key ="3ef39236f28c42ad85ee93369cc454e8"
openai_api_key = os.environ.get('OPENAI_API_KEY')
print("openai version",openai.__version__)
# import getpass

MONGODB_ATLAS_CLUSTER_URI = 'dummy mongo uri'
from pymongo import MongoClient

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "legal"
COLLECTION_NAME = "legal_contract_document"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

os.environ['OPENAI_API_KEY'] ="3ef39236f28c42ad85ee93369cc454e8"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://pn-llm-1.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

PRIMARY_COLOR = "#033797"
SECONDARY_COLOR = "#dd5a0c"
BACKGROUND_COLOR = "#ffffff"
TEXT_COLOR = "#033797"
 
import boto3
from botocore.exceptions import ClientError


import requests

url_retrieval = "http://127.0.0.1:8001/get_response"
url_vectorization_classify = "http://127.0.0.1:8000/classify"
url_vectorization_upload = "http://127.0.0.1:8000/upload"


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)



# def get_response(query,pdf_qa_tup,chat_history):
#     bucket_name = 'contract-bucket-rag'
#     pdf_qa=pdf_qa_tup
#     print('-------before pdf_qa-------')
#     result = pdf_qa(
#         {"question": query, "chat_history": chat_history})
#     print('-------after pdf_qa-------')
#     return result["answer"]

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


if __name__ == "__main__":

    st.set_page_config(
        page_title="PeritusHub's DAS demo",
        page_icon="🤖",
        layout="wide")
    

    st.title("Welcome to PeritusHub's Document Analysis Chatbot!")  # Brand-consistent language
    if "pdf_processed" not in st.session_state:
        st.session_state["pdf_processed"] = False

    if "results" not in st.session_state:
        st.session_state["pdf_processed"] = []
   
    pdf_file = st.file_uploader(":grey[Choose a PDF file]", type="pdf")

    s3 = boto3.resource('s3')
    bucket_name = 'contract-bucket-rag'

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": '''Hi I am a Document Analysis bot, How can I help you today?\n
             '''}]
    #save PDF to the s3 
    if pdf_file is not None and not st.session_state["pdf_processed"]:
        # Define Store Name
        store_name = pdf_file.name[:-4]
        s3_file = f"{store_name}.pdf"
        s3 = boto3.resource('s3')
        bucket_name = 'contract-bucket-rag'
        print("pdf_file",pdf_file,"filename",pdf_file.name)
        # Upload the file to S3 and get the URL
        #s3_file_url = upload_file_to_s3(pdf_file, bucket_name)
        response_url = requests.post(url_vectorization_upload, file=pdf_file)
        print('S3 File URL:', response_url)
        #text_data = extract_text_from_pdf(pdf_file)
        #print(len(text_data))
        response_classify = requests.post(url_vectorization_classify, file=pdf_file)
        classification_result,summary_result,extraction_result = response_classify(0,1,2)
        st.session_state['results'] = [classification_result,summary_result,extraction_result]
        st.markdown('##### **Classification**')
        st.markdown(f'<small>{classification_result}</small>', unsafe_allow_html=True)
        
        #summary_result = summarize_text(text_data)
        st.markdown('##### **Summarization**')
        st.markdown(f'<small>{summary_result}</small>', unsafe_allow_html=True)
        
        st.markdown('##### **Extraction**')
        for key, value in extraction_result.items():
        #extraction_result = extract_information(text_data)
            if key != 'party_metadata' and key != 'date' and key != 'contract_type' and key !='summary':
                st.markdown(f'<small> **{key}** : {value} </small> ', unsafe_allow_html=True)

            if key == 'date':
                for key in extraction_result['date']:
                    st.markdown(f'<small> **{key}** : {value[key]} </small> ', unsafe_allow_html=True)
        st.session_state["pdf_processed"] = True
    elif "results" in st.session_state and len(st.session_state['results']) != 0:
        classification_result,summary_result,extraction_result = st.session_state['results']
        st.markdown('##### **Classification**')
        st.markdown(f'<small>{classification_result}</small>', unsafe_allow_html=True)

        st.markdown('##### **Summarization**')
        st.markdown(f'<small>{summary_result}</small>', unsafe_allow_html=True)
        #data = json.loads(extraction_result)
        st.markdown('##### **Extraction**')
        for key, value in extraction_result.items():
        #extraction_result = extract_information(text_data)
            if key != 'party_metadata' and key != 'date' and key != 'contract_type' and key !='summary':
                st.markdown(f'<small> **{key}** : {value} </small> ', unsafe_allow_html=True)
            if key == 'date':
                for key in extraction_result['date']:
                    st.markdown(f'<small> **{key}** : {value[key]} </small> ', unsafe_allow_html=True)


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    messages = st.session_state.messages

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"]=[]
    chat_history=st.session_state.get("chat_history")


    if user_input := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("PeritusHub's bot is thinking ..."):
                data = {"user_input": user_input,"chat_history": chat_history}
                assistant_response = requests.post(url_retrieval, json=data)
                st.session_state["chat_history"]=chat_history[-10:]
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state["response"] = full_response    
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Get the last user message and bot response from the chat history
    user_input = st.session_state.messages[-2]['content'] if len(st.session_state.messages) >= 2 else ""
    bot_response = st.session_state.messages[-1]['content'] if len(st.session_state.messages) >= 1 else ""

