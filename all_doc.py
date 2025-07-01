import pandas as pd
from langchain.schema import Document
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from typing import Sequence, Annotated, TypedDict
from operator import add as add_messages

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# -----------------------------------
# Load and clean the Excel file
# -----------------------------------
def load_cleaned_excel(path):
    return pd.read_excel(path).fillna(0)

medical_data = load_cleaned_excel(r"Data\150_SET_cleaned.xlsx")

# -----------------------------------
# Create Documents from Excel rows
# -----------------------------------
def convert_rows_to_documents(df):
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(page_content=content, metadata={})
        documents.append(doc)
    return documents

documents = convert_rows_to_documents(medical_data)



for idx, doc in enumerate(documents):
    print(f"Document {idx+1}: {doc}")