##medical data rag.
💊🩺 Scenario: Multi-Agent Medical Assistant in a Hospital
You're designing a hospital chatbot assistant that routes internal staff queries to the appropriate specialist:

🏥 Scenario Description:
Hospital staff (nurses, doctors, pharmacists) interact with a shared AI assistant for different tasks:
🧑‍⚕ Nurse messages:
  Image “Can I give 500mg of paracetamol to a child weighing 15kg?”
​
This needs a pharmacist to answer based on dosage safety.

🩺 Doctor messages:
  Image “Show me the latest lab results for Mr. Sharma.”
“What’s the diagnosis summary for this patient?”
​
This needs a doctor agent who can understand diagnostic data or fetch results.

💉 Pharmacist messages:
  Image “This patient is allergic to penicillin. What’s the best alternative antibiotic?”
​
This needs a collaborative response, potentially by both doctor and pharmacist agents.

🎯 Your Task:
Alter the code so that:
	•	The classifier detects if the message is for the doctor, nurse, or pharmacist.
	•	Routes the message to the correct agent.
	•	Each agent gives a specialized response based on their domain.

🧠 Why this helps:
In real-world hospital settings, communication among departments is complex. Automating and classifying them improves:
	•	Accuracy (correct domain agent responds)
	•	Time-efficiency (no need to redirect manually)
	•	Safety (ensures the right expertise handles the request)