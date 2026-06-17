import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq

# 1. Initialize Embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

DB_DIR = "./chroma_db"
DOCS_DIR = "./knowledge_base"

# Ensure the knowledge base folder exists
os.makedirs(DOCS_DIR, exist_ok=True)

# 2. PERSISTENT VECTOR STORE INITIALIZATION
# This runs ONLY ONCE when app.py starts, not every time a user scans a leaf.
if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("🧠 --- Loading existing RAG Brain from disk (Instant) ---")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
else:
    print("📚 --- First-time setup: Reading PDFs and building RAG Brain (May take a minute) ---")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()
    
    if not documents:
        print("⚠️ WARNING: No PDFs found in knowledge_base folder!")
        # Fallback to an empty vector store to prevent crashes
        vector_store = None 
    else:
        # Better chunking to capture full agricultural context
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(docs, embedding_function, persist_directory=DB_DIR)
        print(f"✅ --- Brain built successfully with {len(docs)} chunks! ---")


# 3. THE INSTANT QUERY ENGINE
def get_rag_advice(farm_name, crop_type, moisture, ndvi):
    if not vector_store:
        return "Knowledge base is empty. Please add PDFs to the 'knowledge_base' folder and restart the server."

    # Retrieve top 5 most relevant paragraphs (fixed Context Blindness)
    query = f"{crop_type} crop farming advice regarding disease, soil moisture {moisture}%, and NDVI {ndvi}."
    relevant_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in relevant_docs])

    # Generate grounded advice
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    system_prompt = f"""
    You are an expert Indian Agronomist AI. 
    Use ONLY the following research context to provide highly actionable advice. 
    If the context does not contain the answer, state that clearly rather than guessing.
    
    --- RESEARCH CONTEXT ---
    {context}
    ------------------------
    
    Task: Analyze Farm '{farm_name}' (Crop: {crop_type}) with Soil Moisture {moisture}% and Satellite NDVI {ndvi}.
    Give concise, 3-sentence actionable advice based on the context above.
    """
    
    response = llm.invoke(system_prompt)
    return response.content