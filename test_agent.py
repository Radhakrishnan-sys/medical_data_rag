from langchain_core.messages import HumanMessage

from difflib import SequenceMatcher

from excel_app import rag_agent

from fpdf import FPDF
 
# Test cases

test_cases = [

    {

        "prompt": "what is ANTONIO PEREIRA 's age and gender?",

        "expected": "ANTONIO PEREIRA is 53 years old Male "

    },

    {

        "prompt": "What is the TSH value of G02434/12?",

        "expected": "The TSH value of art no.G02434/12 is 3.92"

    },

    {

        "prompt": " what is the FBSL value of art no. G03715/15?",

        "expected": " the FBSL value of art no. G03715/15 is 105"

    }

    # Add more test cases here

]
 
def similarity(a, b):

    """Calculate the similarity between two strings."""

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
 
def run_tests(rag_agent, threshold=0.85, pdf_filename="test_report.pdf"):

    print("\n=== Running Test Cases ===\n")

    passed = 0
 
    # PDF setup

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)

    pdf.cell(0, 10, "Test Report", ln=1, align="C")

    pdf.ln(5)

    pdf.set_font("Arial", size=12)
 
    for idx, test in enumerate(test_cases, 1):

        prompt = test["prompt"]

        expected = test["expected"]
 
        result = rag_agent.invoke({"messages": [HumanMessage(content=prompt)]})

        response_msg = next((msg for msg in reversed(result["messages"]) if hasattr(msg, "content")), None)
 
        response = response_msg.content.strip() if response_msg else "No response"

        sim_score = similarity(response, expected)
 
        # Terminal output (with emojis)

        print(f"\nTest Case {idx}: {prompt}")

        print(f"Expected: {expected}")

        print(f"Actual:   {response}")

        print(f"Similarity Score: {sim_score:.2f}")
 
        if sim_score >= threshold:

            print("✅ Test Passed")

            result_str = "Test Passed"   # No emoji for PDF

            passed += 1

        else:

            print("❌ Test Failed")

            result_str = "Test Failed"   # No emoji for PDF
 
        # PDF output (plain text only)

        pdf.set_font("Arial", 'B', 12)

        pdf.cell(0, 10, f"Test Case {idx}:", ln=1)

        pdf.set_font("Arial", '', 12)

        pdf.multi_cell(0, 8, f"Prompt: {prompt}")

        pdf.multi_cell(0, 8, f"Expected: {expected}")

        pdf.multi_cell(0, 8, f"Actual: {response}")

        pdf.cell(0, 8, f"Similarity Score: {sim_score:.2f}", ln=1)

        pdf.cell(0, 8, f"{result_str}", ln=1)

        pdf.ln(4)
 
    total = len(test_cases)

    accuracy = passed / total * 100

    print(f"\n=== Test Summary ===")

    print(f"Passed {passed} / {total} tests")

    print(f"Accuracy: {accuracy:.2f}%")
 
    # PDF summary

    pdf.set_font("Arial", 'B', 12)

    pdf.cell(0, 10, "Test Summary", ln=1)

    pdf.set_font("Arial", '', 12)

    pdf.cell(0, 8, f"Passed {passed} / {total} tests", ln=1)

    pdf.cell(0, 8, f"Accuracy: {accuracy:.2f}%", ln=1)
 
    pdf.output(pdf_filename)

    print(f"\nPDF report saved as {pdf_filename}")
 
if __name__ == "__main__":

    run_tests(rag_agent)

 