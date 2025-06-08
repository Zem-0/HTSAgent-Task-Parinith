from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os

# ✅ Set your Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD-dKPHWu7IfAronnf6b--GweWiNl-eGxk"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load saved FAISS DB
def load_vector_db(path="pdf_vector_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        path, 
        embeddings,
        allow_dangerous_deserialization=True  # Only use if you trust the source of the vectorstore
    )
    return vectorstore

# ✅ Set up Gemini + RAG chain
def setup_chat_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ✅ Chat loop
def chat_with_pdf():
    vectorstore = load_vector_db()
    rag_chain = setup_chat_chain(vectorstore)

    print("\n💬 Ask questions about your PDF (type 'exit' to quit):")
    while True:
        query = input("🔎 You: ")
        if query.lower() in ['exit', 'quit']:
            break
        result = rag_chain(query)
        print("\n📘 Gemini:\n", result['result'], "\n")

# ✅ Run chat
if __name__ == "__main__":
    chat_with_pdf()