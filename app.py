import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. API KEY SETUP
# It will look for GOOGLE_API_KEY in your Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.error("❌ Please add GOOGLE_API_KEY to your Streamlit Secrets!")
    st.stop()

# 2. CONFIGURE GEMINI MODELS
# We use 'gemini-1.5-flash' because it's fast and free
Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=api_key)

st.set_page_config(page_title="Lecture AI (Gemini)", layout="centered")
st.title("📚 My Lecture AI Tutor (Powered by Gemini)")

# 3. LOADING DATA
@st.cache_resource(show_spinner="Reading your PDFs...")
def initialize_index():
    if not os.path.exists("./storage"):
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist()
        return index
    else:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        return load_index_from_storage(storage_context)

index = initialize_index()
query_engine = index.as_query_engine()

# 4. UI FEATURES
mode = st.sidebar.radio("Go to:", ["Chat", "Summarizer", "Quiz Maker"])

if mode == "Chat":
    user_q = st.text_input("Ask anything about your lectures:")
    if user_q:
        response = query_engine.query(user_q)
        st.write(response.response)

elif mode == "Summarizer":
    if st.button("Generate Study Notes"):
        response = query_engine.query("Summarize the main topics into bullet points.")
        st.write(response.response)

elif mode == "Quiz Maker":
    if st.button("Generate 3 MCQs"):
        response = query_engine.query("Create 3 multiple choice questions based on the PDFs.")
        st.write(response.response)
