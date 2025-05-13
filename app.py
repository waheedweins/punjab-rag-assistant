import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load Gemini API key from secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# LLM and Embedding setup
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load and split documents
@st.cache_resource
def load_chunks():
    file_paths = ["leave_rules_rag_app.pdf", "rules_rag_app.pdf"]
    loaders = [PyPDFLoader(path) for path in file_paths]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

chunks = load_chunks()

# Vector retriever setup
@st.cache_resource
def get_retriever():
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3})

retriever = get_retriever()

# QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# UI
st.title("📘 Punjab Employees RAG Assistant")
st.markdown("Ask about service rules, leave, or policies.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("👤 You:", placeholder="e.g., What are the maternity leave rules?")

if query:
    st.session_state.chat_history.append(HumanMessage(content=query))
    result = qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
    answer = result["answer"]
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.markdown(f"🧠 **Assistant:** {answer}")
