"""Python file to serve as the frontend"""
import sys
import os
sys.path.append(os.path.abspath('.'))
import streamlit as st
import time

from langchain.chains import ConversationChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
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
import tempfile
from pymongo import MongoClient

openai.api_type = "azure"
openai.api_base = "https://pn-llm-1.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key ="3ef39236f28c42ad85ee93369cc454e8"
openai_api_key = os.environ.get('OPENAI_API_KEY')
print("openai version",openai.__version__)
# import getpass

MONGODB_ATLAS_CLUSTER_URI = 'mongodb+srv://gaurav_peritushub:4fx7xenS71Ow8hSI@cluster0-poc1.083qk.mongodb.net/'

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

s3 = boto3.resource('s3')
bucket_name = 'contract-bucket-rag'


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


def load_chain():
    """Logic for loading the chain you want to use should go here."""
            # If question is another language translate to english
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    qa_template = """You are a legal expert assistant who assist users for extracting important information from contracts based on the context provided, 
        Provide answer to the user queries.
        When providing an answer, use tone like a helpful assistant.
        Extract right facts with correct spelling
        Question: {question}
        =========
        {context}
        =========
        """
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
                                    MONGODB_ATLAS_CLUSTER_URI,
                                    DB_NAME + "." + COLLECTION_NAME,
                                    OpenAIEmbeddings(deployment="embeddings"),
                                    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
                                )
    
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(engine="patentLearn",temperature=0.0, model_name='gpt-4'),#gpt-3.5-turbo
        retriever=vector_search.as_retriever(search_type="similarity",search_kwargs={"k": 5},),
        return_source_documents=True,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        get_chat_history=get_chat_history,
        verbose=False,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT},
    )
    return pdf_qa



def get_response(query,pdf_qa_tup,chat_history):

    pdf_qa=pdf_qa_tup
    print('-------before pdf_qa-------')
    result = pdf_qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    print('-------after pdf_qa-------')
    return result["answer"]        


