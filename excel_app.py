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


def row_to_text(row):
    return (
        f"SR NO: {row['SR NO']}, "
        f"Date of Confirmed Positive: {row['DATE OF CONFIRMED POSITIVE']}, "
        f"White Count: {row['white ']}, "
        f"Patient Name: {row['NAME']}, "
        f"Age: {row['AGE']}, "
        f"Sex: {row['SEX']}, "
        f"HIV 1/2: {row['HIV 1/2']}, "
        f"Baseline CD4: {row['baseline cd4']}, "
        f"CD4 COUNT: {row['CD4 COUNT']}, "
        f"Improvement in CD4: {row['IMPROVEMENT IN CD4']}, "
        f"ART NO.: {row['ART NO.']}, "
        f"Duration of HIV: {row['DURATION HIV']}, "
        f"Start of ART: {row['START OF ART']}, "
        f"Started ART Type: {row[' STARTED ART TYPE']}, "
        f"HAART Type in Use: {row['HAART TYPE IN USE']}, "
        f"Baseline Viral Load: {row['BASELINE VIRAL LOAD']}, "
        f"Viral Load: {row['VIRAL LOAD ']}, "
        f"HAART Duration: {row['HAART DURATION']}, "
        f"Creatinine (CR.): {row['CR.']}, "
        f"FBSL: {row['FBSL']}, "
        f"FT3: {row['FT3']}, "
        f"FT4: {row['FT4']}, "
        f"TT3: {row['TT3']}, "
        f"TT4: {row['TT4']}, "
        f"TSH: {row['TSH']}, "
        f"Diagnosis: {row['DIAGNOSIS ']}, "
        f"ANTI TPO: {row['ANTI TPO']}, "
        f"TCHOL: {row['TCHOL']}, "
        f"TG: {row['TG']}, "
        f"HDL: {row['HDL']}, "
        f"LDL: {row['LDL']}, "
        f"VLDL: {row['VLDL']}, "
        f"HB: {row['HB']}, "
        f"TC: {row['TC']}, "
        f"N: {row['N']}, "
        f"L: {row['L']}, "
        f"M: {row['M']}, "
        f"E: {row['E']}, "
        f"B: {row['B']}, "
        f"PCV: {row['PCV']}, "
        f"PLT: {row['PLT']}, "
        f"HBSAG: {row['HBSAG']}, "
        f"ANTI HCV: {row['ANTI HCV']}."
    )

def convert_rows_to_documents(df):
    documents = []
    for _, row in df.iterrows():
        # Convert each row to a text representation 
        text = row_to_text(row)
        doc = Document(page_content=text, metadata={})
        documents.append(doc)
    return documents

documents = convert_rows_to_documents(medical_data)

# -----------------------------------
# Create Chroma Vectorstore
# -----------------------------------
def create_chroma_vectorstore(documents, model_name="all-MiniLM-L6-v2"):
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma.from_documents(documents, embedding=embedder)
    return vectorstore

vectorstore = create_chroma_vectorstore(documents)
retriever = vectorstore.as_retriever()




# Show the first 3 chunks
for i, doc in enumerate(documents[:3]):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print("\n")

print(f"Total chunks created: {len(documents)}")


# -----------------------------------
# Create ChatOpenAI LLM
# -----------------------------------
def create_chat_openai_llm(model_name="gpt-3.5-turbo"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    return ChatOpenAI(model_name=model_name, openai_api_key=api_key)

llm = create_chat_openai_llm()

# -----------------------------------
# Create RetrievalQA Chain
# -----------------------------------
def create_retrieval_qa_chain(llm, retriever, chain_type="stuff", k=150):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
    )
    return qa_chain

qa_chain = create_retrieval_qa_chain(llm, retriever)
print("RetrievalQA chain created successfully.")

# =====================
# STEP 3 - Define Tool
# =====================
@tool
def hospital_retriever_tool(query: str) -> str:
    """
    Search hospital records and return only relevant data.
    If nothing is found, explicitly say so.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant hospital records found."
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])

# =====================
# STEP 4 - Setup Agent
# =====================
tools = [hospital_retriever_tool]
tools_dict = {tool.name: tool for tool in tools}

llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0
system_prompt = """
You are a highly accurate and cautious AI medical assistant specialized in answering questions based on hospital patient records stored in tabular format (e.g., Excel or CSV files).

The patient records contain fields such as:
- Name, ART NO., Age, Sex, CD4 COUNT, Diagnosis, TSH, FBSL, Viral Load, etc.

Your knowledge comes **only** from the retrieved medical documents. These are structured row-wise, where each row represents one patient and includes all their medical values.

**Your Instructions:**
1. Do NOT guess or infer answers outside the documents.
2. Match values exactly using unique identifiers like ART NO., SR NO., or patient names.
3. If a question asks for a value like "What is the TSH value of ART NO. G02434/12?", look for a row where `ART NO.` matches exactly, and then extract the `TSH` value.
4. If no such row or field exists, respond with: **"No relevant hospital records found."**
5. You are allowed to call tools (e.g., retrievers) multiple times to gather better context.
6. Prefer factual, structured responses. Keep answers short and clinically accurate.

DO NOT fabricate information.
DO NOT rely on general medical knowledge.
Search for values by exact matches of ART NO., SR NO., or NAME where needed.
If the query mentions an ART NO., match the record based on the ART NO. field.
Answer ONLY based on the retrieved patient records.
"""


def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        query = t["args"].get("query", "")
        tool_name = t["name"]
        print(f"\nCalling tool: {tool_name} with query: {query}")
        if tool_name not in tools_dict:
            result = "Invalid tool call"
        else:
            result = tools_dict[tool_name].invoke(query)
        results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result)))
    return {"messages": results}

# =====================
# STEP 5 - Build LangGraph
# =====================
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# =====================
# STEP 6 - Run CLI Agent
# =====================
def running_agent():
    print("\n=== HOSPITAL RAG AGENT ===")
    print("Type 'exit' or 'quit' to stop.")

    chat_history = []

    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye.")
            break

        chat_history.append(HumanMessage(content=user_input))
        result = rag_agent.invoke({"messages": chat_history})
        response_msg = next((msg for msg in reversed(result["messages"]) if hasattr(msg, "content")), None)

        if response_msg:
            print("\nAssistant:\n" + response_msg.content)
            chat_history.append(response_msg)

if __name__ == "__main__":
    running_agent()
