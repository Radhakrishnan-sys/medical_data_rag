import pandas as pd
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the Excel file (change path if needed)
df = pd.read_excel("Data/150_SET_cleaned.xlsx")

# Allowed safe functions for eval
SAFE_GLOBALS = {
    "df": df,
    "str": str,
    "int": int,
    "float": float,
    "len": len,
    "__builtins__": {}
}


def parse_query_to_code(question: str) -> str:
    """Use OpenAI to turn a natural language question into a safe pandas expression."""
    prompt = f"""
You are an assistant that converts medical data questions into Python pandas queries.

The DataFrame is named `df` and includes columns like: {', '.join(df.columns[:10])}...

Use `.astype(str).str.upper()` when comparing string fields to avoid errors. 
The DataFrame is called `df`, and all column names are in uppercase.
Only use the following exact column names when writing code:

{', '.join(df.columns)}

Always match columns exactly (case-sensitive), like: "FTP", not "ftp" or "Ftp".

Examples:
- How many male patients? → df[df["SEX"] == "M"].shape[0]
- TSH value for GOVIND SWAMI → df[df["NAME"] == "GOVIND SWAMI"]["TSH"].values[0]

Now, write the correct Python code for:

Question: {question}
Only return the pandas code.
    """

    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip("`").strip()


def answer_query(question: str) -> str:
    """Main logic to generate answer using GPT + pandas eval."""
    print("\n🔍 Interpreting your query with GPT...")

    try:
        pandas_code = parse_query_to_code(question)
        print("🧠 GPT →", pandas_code)

        if "i'm sorry" in pandas_code.lower():
            return "❌ That doesn't look like a valid data question."

        result = eval(pandas_code, SAFE_GLOBALS)
        return result if result is not None else "⚠️ No result found."

    except Exception as e:
        return f"❌ Error: {e}"


def start_cli():
    print("🤖 Hybrid LLM + Structured Medical QA Agent")
    print("Ask questions from Excel (150_SET_cleaned.xlsx). Type 'exit' to quit.\n")

    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting...")
            break

        answer = answer_query(user_input)
        print("📋 Answer:", answer)


if __name__ == "__main__":
    start_cli()
