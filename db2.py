import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from pathlib import Path

# Step 1: Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD-dKPHWu7IfAronnf6b--GweWiNl-eGxk"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Step 2: Create xls directory if it doesn't exist
xls_folder = Path("xls")
xls_folder.mkdir(exist_ok=True)

# 🗂️ List of filenames
excel_files = ["htsdata (1).csv", "htsdata (2).csv", "htsdata.csv","htsdata (3).csv","htsdata (4).csv",]

# 📚 Gather documents from all sheets
docs = []

for file in excel_files:
    path = xls_folder / file
    try:
        # Use read_csv instead of read_excel for CSV files
        df = pd.read_csv(path)
        print(f"[INFO] Successfully loaded {file}")
        
        for _, row in df.iterrows():
            hts = str(row.get("HTS Number", "")).strip()
            desc = str(row.get("Description", "")).strip()
            rate = str(row.get("General Rate of Duty", "")).strip()

            if hts and desc:
                text = f"HTS: {hts} | Description: {desc} | Duty: {rate}"
                docs.append(Document(page_content=text, metadata={"hts": hts}))
    except FileNotFoundError:
        print(f"[WARNING] File not found: {file}")
        continue
    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {str(e)}")
        continue

if not docs:
    raise ValueError("No documents were processed. Check if CSV files exist in the 'xls' folder.")

# 🤖 Build vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("hts_vector_db")

print("✅ Combined HTS vector DB saved to: ./hts_vector_db/")