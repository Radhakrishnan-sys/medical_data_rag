# Importing Libraries:
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("gpt-4o")

# Structured output parser for classifying role
class RoleClassifier(BaseModel):
    role: Literal[
        "doctor", "nurse", "pharmacist",
        "doctor+pharmacist", "nurse+doctor", "nurse+pharmacist"
    ] = Field(
        ..., 
        description="Classify who should handle the query: a doctor, nurse, pharmacist, or combination."
    )

# State for the chatbot
class state(TypedDict):
    messages: Annotated[list, add_messages]
    role: str | None

# Classification node
def classify_message(state: state):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(RoleClassifier)

    result = classifier_llm.invoke([
        {"role": "system", "content": """
Classify the user's message as one of the following EXACT options:
- 'doctor': if it's about diagnosis, lab results, or summaries.
- 'nurse': if it's about patient care, dosage, or procedures.
- 'pharmacist': if it's about medications, allergies, or drug alternatives.
- 'doctor+pharmacist': if both diagnosis and medication advice are needed.
- 'nurse+pharmacist': if care overlaps with medication (e.g., dosage safety).
- 'nurse+doctor': if care + diagnosis clarification is involved.

ONLY return one of these exact strings as your answer.
"""}, 
        {"role": "user", "content": last_message.content}
    ])
    return {"role": result.role}

# Doctor agent
def doctor_agent(state: state):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": """You are a knowledgeable doctor assistant.
Respond to queries involving lab results, diagnoses, and patient history.
Be concise, factual, and medically accurate."""},
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Nurse agent
def nurse_agent(state: state):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": """You are a helpful nurse assistant.
Respond to care-related questions, medication dosages, and procedural guidance.
Ensure clarity and safety in all instructions."""},
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Pharmacist agent
def pharmacist_agent(state: state):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": """You are a pharmacist assistant.
Focus on drug safety, alternatives for allergies, and proper medication handling.
Provide accurate pharmaceutical advice."""},
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Doctor → Pharmacist
def doctor_then_pharmacist(state: state):
    state = doctor_agent(state)
    return pharmacist_agent(state)

# Nurse → Pharmacist
def nurse_then_pharmacist(state: state):
    state = nurse_agent(state)
    return pharmacist_agent(state)

# Nurse → Doctor
def nurse_then_doctor(state: state):
    state = nurse_agent(state)
    return doctor_agent(state)

# Building the graph
graph_builder = StateGraph(state)

# Add all agent nodes
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("doctor", doctor_agent)
graph_builder.add_node("nurse", nurse_agent)
graph_builder.add_node("pharmacist", pharmacist_agent)
graph_builder.add_node("doctor_then_pharmacist", doctor_then_pharmacist)
graph_builder.add_node("nurse_then_pharmacist", nurse_then_pharmacist)
graph_builder.add_node("nurse_then_doctor", nurse_then_doctor)

# Graph edges
graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "classifier",  # changed from "router" to "classifier"
    lambda state: state.get("role"),
    {
        "doctor": "doctor",
        "nurse": "nurse",
        "pharmacist": "pharmacist",
        "doctor+pharmacist": "doctor_then_pharmacist",
        "nurse+pharmacist": "nurse_then_pharmacist",
        "nurse+doctor": "nurse_then_doctor"
    }
)

# End edges
graph_builder.add_edge("doctor", END)
graph_builder.add_edge("nurse", END)
graph_builder.add_edge("pharmacist", END)
graph_builder.add_edge("doctor_then_pharmacist", END)
graph_builder.add_edge("nurse_then_pharmacist", END)
graph_builder.add_edge("nurse_then_doctor", END)

# Compile the graph
graph = graph_builder.compile()

# Run loop
def run_chatbot():
    state = {"messages": [], "role": None}

    while True:
        user_input = input("Message: ")
        if user_input.lower() == "exit":
            print("BYE!")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]
        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()
