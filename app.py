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
# Loaders for each CSV file
# -----------------------------------

def load_csv_as_df(csv_path):
    return pd.read_csv(csv_path)

def load_patient_details(csv_path):
    return load_csv_as_df(csv_path)

def load_diagnosis(csv_path):
    return load_csv_as_df(csv_path)

def load_medications(csv_path):
    return load_csv_as_df(csv_path)

def load_prescriptions(csv_path):
    return load_csv_as_df(csv_path)

def load_alerts(csv_path):
    return load_csv_as_df(csv_path)

def load_diabetic_indices(csv_path):
    return load_csv_as_df(csv_path)

def load_encounters(csv_path):
    return load_csv_as_df(csv_path)

def load_immunizations(csv_path):
    return load_csv_as_df(csv_path)

# Step 2: Load data
patients = load_patient_details(r"Data\patient_details.csv")
diagnosis = load_diagnosis(r"Data\Diagnosis.csv")
medications = load_medications(r"Data\medications.csv")
prescriptions = load_prescriptions(r"Data\prescriptions.csv")
alerts = load_alerts(r"Data\alerts.csv")
indices = load_diabetic_indices(r"Data\diabetic_indices.csv")
encounters = load_encounters(r"Data\encounter_history.csv")
immunizations = load_immunizations(r"Data\immunizations.csv")

# -----------------------------------
# Combiner: build per-patient documents
# -----------------------------------

def combine_patient_documents(
    patient_df,
    diagnosis_df,
    medications_df,
    prescriptions_df,
    alerts_df,
    indices_df,
    encounters_df,
    immunizations_df
):
    patient_docs = []

    for _, patient in patient_df.iterrows():
        pid = patient["PatientID"]
        parts = []

        # Patient details
        parts.append(f"PatientID: {pid}")
        parts.append(f"Name: {patient['Name']}")
        parts.append(f"Sex: {patient['Sex']}")
        parts.append(f"DOB: {patient['DOB']}")
        parts.append(f"Phone: {patient['Phone']}")
        parts.append(f"Address: {patient['Address']}")
        parts.append(f"NextOfKin: {patient['NextOfKin']} ({patient['NextOfKinPhone']}), Address: {patient['NextOfKinAddress']}")

        # Diagnoses
        diag = diagnosis_df[diagnosis_df["PatientID"] == pid]
        if not diag.empty:
            parts.append("Diagnoses:")
            for _, row in diag.iterrows():
                parts.append(f" - {row['Diagnosis']} (State: {row['State']}, Status: {row['Status']})")

        # Medications
        meds = medications_df[medications_df["PatientID"] == pid]
        if not meds.empty:
            parts.append("Medications:")
            for _, row in meds.iterrows():
                parts.append(f" - {row['Medication']} on {row['Date']}")

        # Prescriptions
        presc = prescriptions_df[prescriptions_df["PatientID"] == pid]
        if not presc.empty:
            parts.append("Prescriptions:")
            for _, row in presc.iterrows():
                parts.append(f" - {row['Prescription']}: {row['Instructions']} ({row['Date']})")

        # Alerts
        alerts = alerts_df[alerts_df["PatientID"] == pid]
        if not alerts.empty:
            parts.append("Alerts:")
            for _, row in alerts.iterrows():
                parts.append(f" - {row['Alert']}")

        # Diabetic Indices
        indices = indices_df[indices_df["PatientID"] == pid]
        if not indices.empty:
            parts.append("Diabetic Indices:")
            for _, row in indices.iterrows():
                parts.append(f" - {row['Index']}: {row['Value']} (Most Recent: {row['MostRecent']})")

        # Encounters
        enc = encounters_df[encounters_df["PatientID"] == pid]
        if not enc.empty:
            parts.append("Encounter History:")
            for _, row in enc.iterrows():
                parts.append(
                    f" - {row['Date']}, {row['Facility']}, {row['Specialty']}, {row['Clinician']}, {row['Reason']} ({row['Type']})"
                )

        # Immunizations
        imm = immunizations_df[immunizations_df["PatientID"] == pid]
        if not imm.empty:
            parts.append("Immunizations:")
            for _, row in imm.iterrows():
                parts.append(f" - {row['Immunization']}: {row['NumberReceived']} doses (Most Recent: {row['MostRecent']})")

        # Combine all parts into one text block
        full_text = "\n".join(parts)

        # Create Document
        doc = Document(page_content=full_text, metadata={"PatientID": pid})
        patient_docs.append(doc)

    return patient_docs


    # Step 3: Combine documents
documents = combine_patient_documents(
        patients, diagnosis, medications, prescriptions,
        alerts, indices, encounters, immunizations
    )


def create_chroma_vectorstore(lc_documents, model_name="all-MiniLM-L6-v2"):
    """
    Create a Chroma vectorstore from LangChain documents using HuggingFace embeddings.

    Args:
        lc_documents (list): List of LangChain Document objects.
        model_name (str): The HuggingFace model to use for embeddings.

    Returns:
        Chroma: A Chroma vectorstore instance.
    """
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma.from_documents(lc_documents, embedding=embedder)
    return vectorstore

vectorstore = create_chroma_vectorstore(documents)
retriever = vectorstore.as_retriever()

def create_chat_openai_llm(model_name="gpt-3.5-turbo"):
    """
    Create a ChatOpenAI LLM instance using API key from .env.

    Args:
        model_name (str): OpenAI model name.

    Returns:
        ChatOpenAI: An LLM instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    return ChatOpenAI(model_name=model_name, openai_api_key=api_key)

llm = create_chat_openai_llm()

def create_retrieval_qa_chain(llm, retriever, chain_type="stuff", k=5):
    """
    Create a RetrievalQA chain.

    Args:
        llm (ChatOpenAI): The LLM instance.
        retriever: The vectorstore retriever.
        chain_type (str): Type of chain ("stuff", "map_reduce", etc.).
        k (int): Number of top documents to retrieve.

    Returns:
        RetrievalQA: A QA chain ready to invoke.
    """
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
You are a medical assistant AI. Answer ONLY using the hospital patient documents available.
If you cannot find the answer in those documents, say: "No relevant hospital records found."
Do NOT guess or fabricate any response.
You are allowed to use tools multiple times if needed.
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
