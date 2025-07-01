import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variable
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your Excel into a DataFrame
df = pd.read_excel("Data/150_SET_cleaned.xlsx").fillna("")
df.columns = [col.strip().upper() for col in df.columns]  # clean column names

# Safe context to evaluate the query
SAFE_GLOBALS = {"df": df, "__builtins__": {}, "str": str }

def parse_query_to_code(question: str) -> str:
    """Use GPT-4 to convert the user's question into a pandas DataFrame query."""
    prompt = f"""
You are a helpful assistant that translates natural language questions into Python pandas DataFrame code.

The DataFrame is named `df` and contains columns like: {', '.join(df.columns[:12])}...

Examples:
- "How many male patients?" → df[df["SEX"] == "M"].shape[0]
- "What is the TSH of GOVIND SWAMI?" → df[df["NAME"] == "GOVIND SWAMI"]["TSH"].values[0]
- "List patients who are HIV negative" → df[df["HIV 1/2"].astype(str).str.upper() == "NEG"][["NAME", "ART NO."]]

Now write the correct pandas code to answer:
"{question}"
(Only return code — no explanation)
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip("`").strip()

def answer_query(question: str):
    print("\n🔍 Interpreting your query with GPT...")
    try:
        pandas_code = parse_query_to_code(question)
        print("🧠 GPT →", pandas_code)
        result = eval(pandas_code, SAFE_GLOBALS)
        return result if result is not None else "No result found."
    except Exception as e:
        return f"❌ Error: {e}"

def run_hybrid_agent():
    print("🤖 Hybrid LLM + Structured Medical QA Agent")
    print("Ask questions from Excel (150_SET_cleaned.xlsx). Type 'exit' to quit.")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        result = answer_query(q)
        print("📋 Answer:", result)

if __name__ == "__main__":
    run_hybrid_agent()
