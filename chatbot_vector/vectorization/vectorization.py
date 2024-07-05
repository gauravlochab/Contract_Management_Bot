"""Python file to serve as the frontend"""
import sys
import os
sys.path.append(os.path.abspath('.'))
import streamlit as st
import time
from flask import Flask, request, jsonify
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from pathlib import Path
from PyPDF2 import PdfReader
import datetime
from urllib.parse import quote

import pandas as pd
import datetime
import pdfplumber
import json
import io
import openai
import tempfile

import uuid
import boto3
from botocore.exceptions import ClientError
from pymongo import MongoClient



openai.api_type = "azure"
openai.api_base = "https://pn-llm-1.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key ="3ef39236f28c42ad85ee93369cc454e8"
openai_api_key = os.environ.get('OPENAI_API_KEY')
print("openai version",openai.__version__)
# import getpass

MONGODB_ATLAS_CLUSTER_URI = 'dummy-uri-mangodb'

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "legal"
COLLECTION_NAME = "legal_contract_document"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

os.environ['OPENAI_API_KEY'] ="dummy-key"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://pn-llm-1.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "dummy-version"


s3 = boto3.resource('s3')
bucket_name = 'dummy-bucket-name'
    


def get_response_openai(file):

    try:
        response = openai.ChatCompletion.create(
                    engine="gpt32k",
                    temperature=0.7,
                    max_tokens=1200,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    seed=0,
                    messages=[
                    {"role": "system", "content": "You are a legal expert"},
                    {"role": "user", "content":f"""Your task is to review and extract important contract information in JSON response based on below strict guidelines

                        Strict Guidelines:
                        1. Response should be strictly in JSON format as per Response Format, make sure not to add any conversational preamble text at start or end of response.
                        2. Make sure you pick the Contract Type of the document from the Contract Type List provided below only.
                        3. If a title is not found in the DOCUMENT, then output title as empty ("")
                        4. If the parties are not found then  mention party_information as empty ("")
                        5. Ensure that your party_type selection must be from the list only : "Seller, Buyer, Disclosing Party, Receiving Party".
                        6. Ensure that "party_information" (sample format ( party names ("references"))) must have party names along with their abbrevation and don\\'t add any quotes.
                        7. Ensure that "purpose_of_contract" should detail the purpose of contract
                        8. Ensure that party_metadata.reference is party abbrevations, if empty then add party_name.
                        9. Ensure that governing_law is just and exact name of governing law of contract, not the clause.
                        10.Ensure that  "role_of_party_in_contract" is stating the role of respective party.
                        11.Ensure that  start_date specify start or effective date of the contract in "dd/mm/yyyy" format.
                        12. Ensure that end_date is termination date of the contract in "dd/mm/yyyy" format.
                        13. Ensure that contract_value ois contract value mentioned in contract document.
                        14. Ensure that "summary" should contain the summary of the document  
                        Please ensure that the "party_type" accurately reflects the roles of the parties as defined by the specific "contract_type." For confidentiality agreements or NDA, the roles are "Disclosing Party" and "Receiving Party," while for sale and purchase agreements (MSA, Service Agreement), the roles are "Seller" and "Buyer."

                        Response Format :
                        {{
                                "contract_type": "",
                                "title": "",
                                "purpose_of_contract": "",
                                "governing_law": "",
                                "party_information": "",
                                "contract_value":"",
                                "summary":"",
                                "party_metadata": [
                                        {{
                                                "location": "",
                                                "party_name": "",
                                                "reference": "",
                                                "role_of_party_in_contract": "",
                                                "party_type": []
                                        }}
                                ],
                                "date":
                                {{
                                    "start_date":"",
                                    "end_date":""
                                }}
                        }}
                        
                        Contract Type List :
                        ```
                        contract_type: identify type from one of the CATEGORY of contract DOCUMENT.
                        CATEGORY:
                            "Non Disclosure Agreement"
                            "Code of Conduct"
                            "Data Processing Addendum"
                            "Software as a Service Agreement"
                            "Consulting Agreement"
                            "Facility Agreement"
                            "General Commercial Agreement"
                            "License Agreement"
                            "Master Service Agreement"
                            "Real Estate Agreement"
                            "Purchase Agreement"
                            "Purchase Order"
                            "Employment Agreement"
                            "Data Processing Agreement"
                            "Sales Agency Agreement"
                            "Service Agreement Contract"
                            "Distribution Agreement"
                            "Contract Extension Letter"
                            "Supply of Goods Agreement"
                            or "Other"
                        ```
                        
                        The Contract Document is as follows:
                        ```
                        {file}
                        ```
                                    
                        """ },
                        ]
                    )

    except Exception as e:
        print("Error in creating sections  from openAI:")
        raise BaseException("Error in creating sections  from openAI:")
    try:
        response_list = [response['choices'][0]['message']['content']]
        return response_list
    except Exception as e:
        print("OpenAI Response (Streaming) Error: " )
        raise BaseException("OpenAI Response (Streaming) Error: " )


def extract_text_from_pdf(uploaded_file):
    #print('text aane se pehle',uploaded_file)
    with pdfplumber.open(uploaded_file) as pdf:
        #print('text aara hai')
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        #print('text aane ke baad')    
    return text          

def process_pdf(file_path):
    #print(file_path)
    documents = []
    loader = PyPDFLoader(file_path)
    documents.extend(loader.load())
    return documents

def process_file(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
            #print(file.getvalue())
            temp.write(file.getvalue())
            temp_file_path = temp.name
    documents = []
    loader = PyPDFLoader(temp_file_path)
    documents.extend(loader.load())
    #print("Muthi maaro doobara")
    os.remove(temp_file_path)
    return documents

def upload_file_to_s3(pdf_file, bucket_name,filename):
    #print(type(pdf_file))
    file_name = pdf_file.filename
    #print(file_name)
    bucket = s3.Bucket(bucket_name)
    safe_file_name = file_name.replace(' ', '+')
    # Check if file already exists in S3
    objs = list(bucket.objects.filter(Prefix=file_name))
    #print(objs)
    if len(objs) > 0 and objs[0].key == file_name:
        print('File already exists in S3. Not uploading.')
    else:
        # If file does not exist, upload it
        bucket.put_object(Key=file_name, Body=pdf_file.getvalue())
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(pdf_file.getvalue())
            temp_file_path = temp.name
        print('File uploaded to S3.')    
        text = process_pdf(temp_file_path)
        os.remove(temp_file_path)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(text)
        for idx, text in enumerate(documents):
            documents[idx].metadata['file_name'] = filename
        #print('doc chunked')
        #print(documents[0])
        vector_search = MongoDBAtlasVectorSearch.from_documents(
                            documents=documents,
                            embedding=OpenAIEmbeddings(deployment="embeddings"),
                            collection=MONGODB_COLLECTION,
                            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
                        )

        print('vectorization done in MongoDB')
        safe_file_name = file_name.replace(' ', '+')  # Replace spaces with '+'
        print('safe_file_name',safe_file_name)
    return f"https://{bucket_name}.s3.us-east-2.amazonaws.com/{safe_file_name}"    


def upload_to_mongodb(response_json,filename):
    # Connect to MongoDB
    client = MongoClient('mongodb+srv://gaurav_peritushub:4fx7xenS71Ow8hSI@cluster0-poc1.083qk.mongodb.net/')
    db = client['legal']
    collection = db['legal_meta_data']
    doc_id = uuid.uuid4()
    document = {
        "file_name": filename,
        "extraction": json.loads(response_json[0]),
        "_id": str(doc_id)  # convert UUID to string before storing
    }
    # Insert the response into the MongoDB collection
    collection.insert_one(document)
