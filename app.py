import streamlit as st
# Set page config first!
st.set_page_config(page_title="HTS & PDF RAG Assistant", layout="wide")

# Now import other dependencies
import pandas as pd
import re
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os

# ---------------------- CONFIG ---------------------- #
os.environ["GOOGLE_API_KEY"] = "AIzaSyD-dKPHWu7IfAronnf6b--GweWiNl-eGxk"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
MODEL = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# ------------------ Load Vector DBs ------------------ #
@st.cache_resource
def load_vectorstores():
    hts_db = FAISS.load_local("hts_vector_db", EMBEDDINGS, allow_dangerous_deserialization=True)
    pdf_db = FAISS.load_local("pdf_vector_db", EMBEDDINGS, allow_dangerous_deserialization=True)
    return hts_db, pdf_db

hts_vectorstore, pdf_vectorstore = load_vectorstores()

hts_qa = RetrievalQA.from_chain_type(llm=MODEL, retriever=hts_vectorstore.as_retriever())
pdf_qa = RetrievalQA.from_chain_type(llm=MODEL, retriever=pdf_vectorstore.as_retriever())

# ------------------ Duty Calculator ------------------ #
def parse_duty(duty_str, cif_value, unit_weight=None, quantity=None):
    if pd.isna(duty_str) or duty_str.strip() == "":
        return 0.0
    duty_str = duty_str.strip().lower()
    if "free" in duty_str:
        return 0.0
    if match := re.search(r"([\d.]+)\s*%", duty_str):
        return float(match.group(1)) / 100
    if match := re.search(r"([\d.]+)\s*¢/kg", duty_str):
        return (float(match.group(1)) * unit_weight) / (100 * cif_value) if unit_weight else 0.0
    if match := re.search(r"\$([\d.]+)/unit", duty_str):
        return (float(match.group(1)) * quantity) / cif_value if quantity else 0.0
    return 0.0

def calculate_duty(hts_code, product_cost, freight, insurance, quantity, unit_weight):
    df = pd.read_csv("htsdata.csv")
    row = df[df["HTS Number"].astype(str).str.strip() == str(hts_code).strip()]
    if row.empty:
        return None, "HTS code not found in the dataset."
    cif = product_cost + freight + insurance
    duty_str = row.iloc[0]["General Rate of Duty"]
    rate = parse_duty(duty_str, cif, unit_weight, quantity)
    duty_amount = rate * cif
    landed_cost = cif + duty_amount
    return {
        "HTS Code": hts_code,
        "Description": row.iloc[0]["Description"],
        "Duty Rate": duty_str,
        "CIF Value": round(cif, 2),
        "Duty Amount": round(duty_amount, 2),
        "Landed Cost": round(landed_cost, 2)
    }, None

# ------------------ Streamlit UI ------------------ #
st.title("📦 TariffBot: HTS Duty Calculator + PDF RAG Agent")

tabs = st.tabs(["📄 Ask PDF Agent", "📦 HTS Calculator + Chat"])

# ------------------ Tab 1: PDF Chat ------------------ #
with tabs[0]:
    st.subheader("Chat with General Notes PDF")
    question = st.text_input("Ask something about the PDF:")
    if question:
        with st.spinner("Thinking..."):
            response = pdf_qa.run(question)
        st.success(response)

# ------------------ Tab 2: HTS RAG + Calculator ------------------ #
with tabs[1]:
    st.subheader("HTS Lookup + Duty Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### HTS Q&A")
        hts_query = st.text_input("Ask a question about HTS data (e.g., 'Duty for 0201.10.00'):")
        if hts_query:
            with st.spinner("Thinking..."):
                answer = hts_qa.run(hts_query)
            st.success(answer)

    with col2:
        st.markdown("### Duty Calculator")
        with st.form("calc_form"):
            hts_code = st.text_input("HTS Code", value="0201.10.00")
            product_cost = st.number_input("Product Cost (USD)", value=1000.0)
            freight = st.number_input("Freight (USD)", value=100.0)
            insurance = st.number_input("Insurance (USD)", value=50.0)
            quantity = st.number_input("Quantity", value=10)
            unit_weight = st.number_input("Unit Weight (kg)", value=5.0)
            submit = st.form_submit_button("Calculate")
            if submit:
                result, error = calculate_duty(hts_code, product_cost, freight, insurance, quantity, unit_weight)
                if error:
                    st.error(error)
                else:
                    st.json(result)