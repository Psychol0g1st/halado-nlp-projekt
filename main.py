from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- CORS middleware import
from pydantic import BaseModel
import agent as rag_agent_module
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="RAG Agent API",
    description="API for interacting with a Retrieval Augmented Generation agent.",
    version="0.1.0"
)

# Enable CORS for your React frontend
origins = [
    "http://localhost:3000",
    "http://medical_rag_frontend_service:3000",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Initialization ---
PERSIST_DIR_FASTAPI = "./storage"
if not os.path.exists(PERSIST_DIR_FASTAPI) or not os.listdir(PERSIST_DIR_FASTAPI):
    print(f"WARNING: Index directory '{PERSIST_DIR_FASTAPI}' not found or is empty during FastAPI startup.")
    agent_instance = None
else:
    try:
        agent_instance = rag_agent_module.RAGAgent(persist_dir=PERSIST_DIR_FASTAPI)
    except Exception as e:
        print(f"Error initializing RAGAgent: {e}")
        agent_instance = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_info: str = "Retrieved from knowledge base"

@app.on_event("startup")
async def startup_event():
    if agent_instance is None and (not os.path.exists(PERSIST_DIR_FASTAPI) or not os.listdir(PERSIST_DIR_FASTAPI)):
        print("Startup: RAG Agent could not be initialized because the index is missing or empty.")
    elif agent_instance is None:
        print("Startup: RAG Agent could not be initialized due to an error during setup.")
    else:
        print("Startup: RAG Agent initialized successfully.")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="RAG Agent is not available or not initialized. Check server logs.")
    try:
        answer = agent_instance.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your question: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    frontend_dir = "frontend"
    index_html_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_html_path):
        return HTMLResponse(content="<html><body><h1>Frontend not found</h1><p>Ensure index.html is in the 'frontend' directory.</p></body></html>", status_code=404)
    with open(index_html_path) as f:
        return HTMLResponse(content=f.read())
