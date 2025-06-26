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





for idx, doc in enumerate(documents):
    print(f"Document {idx+1}: {doc}")