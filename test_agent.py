from langchain_core.messages import HumanMessage
from difflib import SequenceMatcher
from app import rag_agent

# Import your agent setup from the main module if needed
# from your_module_name import rag_agent

test_cases = [
    {
        "prompt": "What medication did Leslie Brooks received after he got diagnosed from Osteoporosis?",
        "expected": "he received medications such as  ASA 81 mg on 07/2010, Ramipril 10 mg on 06/2006, Amoxicillin 500 mg on 09/2024 "
    },
    {
        "prompt": "What is the date of birth of Debra Palmer?",
        "expected": "The date of birth of debra palmer is DOB: 1988/11/29."
    },
    {
        "prompt": " what is the address, diagnosed details, medications received by PatientID: GME0978 ?",
        "expected": " the address is Address: 18787 Fields Isle, North Danielton, NY 85485. The diagnosed details include Osteoporosis, Coronary Artery Disease , Asthma. the medications received include Medications:"
        "- Lisinopril 10 mg on 03/208"
        "- Clobetasone Cream on 03/2018"
        "- Amoxicillin 500 mg on 04/2006"
        "- Hydrochlorothiazide 25 mg on 04/2020"
        "- Atorvastatin 20 mg on 05/2009"
    }
    # Add more test cases here
]

def similarity(a, b):
    """Calculate the similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def run_tests(rag_agent, threshold=0.85):
    print("\n=== Running Test Cases ===\n")
    passed = 0

    for idx, test in enumerate(test_cases, 1):
        prompt = test["prompt"]
        expected = test["expected"]

        result = rag_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        response_msg = next((msg for msg in reversed(result["messages"]) if hasattr(msg, "content")), None)

        response = response_msg.content.strip() if response_msg else "No response"
        sim_score = similarity(response, expected)

        print(f"\nTest Case {idx}: {prompt}")
        print(f"Expected: {expected}")
        print(f"Actual:   {response}")
        print(f"Similarity Score: {sim_score:.2f}")

        if sim_score >= threshold:
            print("✅ Test Passed")
            passed += 1
        else:
            print("❌ Test Failed")

    print(f"\n=== Test Summary ===")
    print(f"Passed {passed} / {len(test_cases)} tests")
    accuracy = passed / len(test_cases) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_tests(rag_agent)
