import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Updated import
from langchain.docstore.document import Document
import google.generativeai as genai
import os
# ✅ Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyD-dKPHWu7IfAronnf6b--GweWiNl-eGxk"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load and extract text from PDF
def load_pdf_text(file_path):
    print("[INFO] Loading PDF...")
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ Split text into chunks
def split_text(text):
    print("[INFO] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([text])

# ✅ Embed and save to FAISS
def embed_and_save(docs, save_path):
    print("[INFO] Generating embeddings and saving FAISS DB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
    print(f"[SUCCESS] FAISS DB saved at {save_path}/")

# ✅ Main ingest function
def ingest_pdf(pdf_path, save_path="pdf_vector_db"):
    text = load_pdf_text(pdf_path)
    docs = split_text(text)
    embed_and_save(docs, save_path)

# ✅ Call the function
if __name__ == "__main__":
    ingest_pdf("General Notes.pdf")
