import pandas as pd
from dotenv import load_dotenv
import os

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from typing import Sequence, Annotated, TypedDict
from operator import add as add_messages

# === Load Environment ===
load_dotenv()

# === Load and Clean Excel ===
def load_cleaned_excel(path):
    return pd.read_excel(path).fillna("")

medical_data = load_cleaned_excel(r"Data\150_SET_cleaned.xlsx")

# === Convert Each Row into Document with Metadata ===
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
        f"Diagnosis: {row['DIAGNOSIS ']}, "
        f"FBSL: {row['FBSL']}, "
        f"TSH: {row['TSH']}."
    )

def convert_rows_to_documents(df):
    documents = []
    for _, row in df.iterrows():
        text = row_to_text(row)
        doc = Document(
            page_content=text,
            metadata={
                "ART_NO": str(row["ART NO."]).strip(),
                "NAME": str(row["NAME"]).strip().upper(),
                "SEX": str(row["SEX"]).strip().upper(),
                "HIV": str(row["HIV 1/2"]).strip().upper()
            }
        )
        documents.append(doc)
    return documents

documents = convert_rows_to_documents(medical_data)

# === Create Vector Store ===
def create_chroma_vectorstore(documents, model_name="all-MiniLM-L6-v2"):
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma.from_documents(documents, embedding=embedder)
    return vectorstore

vectorstore = create_chroma_vectorstore(documents)

# === Retriever with large k ===
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 150})

# === LLM ===
def create_chat_openai_llm(model_name="gpt-4o"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return ChatOpenAI(model_name=model_name, openai_api_key=api_key)

llm = create_chat_openai_llm()

# === Retrieval QA Chain ===
def create_retrieval_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa_chain = create_retrieval_qa_chain(llm, retriever)

# === Smart Retrieval Tool ===
@tool
def hospital_retriever_tool(query: str) -> str:
    """
    Search hospital records and return only relevant data.
    If nothing is found, explicitly say so.
    """
    query_lower = query.lower()
    
    # Fallback logic for count/list-type queries
    if "how many" in query_lower or "list" in query_lower or "all" in query_lower:
        docs = vectorstore.similarity_search("", k=150)  # return all
    else:
        docs = retriever.invoke(query)

    if not docs:
        return "No relevant hospital records found."
    
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])

# === Tool Bind ===
tools = [hospital_retriever_tool]
tools_dict = {tool.name: tool for tool in tools}
llm = llm.bind_tools(tools)

# === LangGraph Agent State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are a highly accurate and cautious AI medical assistant specialized in answering questions based on hospital patient records stored in Excel format.

Each row represents one patient and includes fields such as:
- Name, ART NO., Age, Sex, CD4 COUNT, Diagnosis, FBSL, TSH, HIV 1/2

**Instructions:**
- Search based on exact matches of ART NO., SR NO., or NAME where needed.
- If the query asks for values like TSH, FBSL, etc., match the right patient using ART NO. or NAME and extract the value.
- For "how many", "list", or "all", retrieve all rows and extract structured facts.
- If no data found, respond with: **"No relevant hospital records found."**
- Do not guess or use external knowledge. Use only retrieved patient records.
"""

# === LLM Call Node ===
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

# === Tool Call Execution ===
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

# === LangGraph ===
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# === Run Agent in CLI ===
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
