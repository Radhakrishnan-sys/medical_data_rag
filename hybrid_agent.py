import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your Excel into a DataFrame
df = pd.read_excel("Data/150_SET_cleaned.xlsx").fillna("")
df.columns = [col.strip().upper() for col in df.columns]  # standardize

# 🔒 Safe eval environment
SAFE_GLOBALS = {"df": df, "__builtins__": {}}

def parse_query_to_code(question: str) -> str:
    """Use GPT-4 to translate user query into pandas code."""
    prompt = f"""
You are a helpful assistant that translates natural language questions into executable pandas DataFrame queries.

The DataFrame is named `df` and contains columns like: {', '.join(df.columns[:10])}...

Only return Python code that extracts the answer, like:
- df[df["SEX"]=="M"].shape[0]
- df[df["NAME"]=="GOVIND SWAMI"]["TSH"].values[0]
- df[df["HIV 1/2"]=="NEG"][["NAME", "ART NO."]]

User question:
{question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message["content"].strip("`").strip()

def answer_query(question: str):
    print("\n🔍 Interpreting your query with GPT...")
    try:
        pandas_code = parse_query_to_code(question)
        print("🧠 GPT →", pandas_code)

        # Run the code in a restricted environment
        result = eval(pandas_code, SAFE_GLOBALS)
        return result if result is not None else "No result found."
    except Exception as e:
        return f"❌ Error: {e}"

# === CLI Loop ===
def run_hybrid_agent():
    print("🤖 Hybrid LLM + Structured Medical QA Agent")
    print("Ask questions from Excel (150_SET_cleaned.xlsx). Type 'exit' to quit.")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        result = answer_query(q)
        print("📋 Answer:", result)

if __name__ == "__main__":
    run_hybrid_agent()
