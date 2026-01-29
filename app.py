import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import os

# 1. API KEY SETUP
# For now, paste your key here. Later, we will hide it for security.
os.environ["OPENAI_API_KEY"] = "AIzaSyB31A-pUwbn279X1GfWFVJB7PDB_TazoMA"

st.set_page_config(page_title="Lecture AI", layout="centered")
st.title("📚 My Lecture AI Tutor")

# 2. LOADING DATA
# This looks into your 'data/' folder
@st.cache_resource(show_spinner="Reading your PDFs...")
def initialize_index():
    if not os.path.exists("./storage"):
        # Read PDFs from your data folder
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # Save this 'memory' so it doesn't have to re-read every time
        index.storage_context.persist()
        return index
    else:
        # Load the memory from the storage folder
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        return load_index_from_storage(storage_context)

index = initialize_index()
query_engine = index.as_query_engine(streaming=True)

# 3. SIDEBAR FEATURES
st.sidebar.header("Study Tools")
mode = st.sidebar.radio("Go to:", ["Chat", "Summarizer", "Quiz Maker"])

if mode == "Chat":
    st.subheader("💬 Ask your Lectures")
    user_q = st.text_input("What would you like to know?")
    if user_q:
        response = query_engine.query(user_q)
        st.write_stream(response.response_gen)

elif mode == "Summarizer":
    st.subheader("📝 Quick Summary")
    if st.button("Summarize all lectures"):
        response = query_engine.query("Summarize the main topics of all uploaded documents into a bulleted list.")
        st.write_stream(response.response_gen)

elif mode == "Quiz Maker":
    st.subheader("🧠 Test Your Knowledge")
    if st.button("Generate 3 Questions"):
        response = query_engine.query("Create 3 multiple choice questions with answers based on the PDFs.")
        st.write_stream(response.response_gen)