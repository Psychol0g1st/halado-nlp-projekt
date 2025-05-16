Okay, this is an interesting project! We'll break it down step-by-step to create the Python file for your LLM agent, the FastAPI backend, a simple frontend, and then package it all with Docker.

First, I need to understand the contents of your Jupyter notebook to see how the LLM agent and vector database are initialized and used. This will be crucial for creating the Python file that encapsulates this logic.

Okay, I've reviewed your Jupyter notebook. This is a comprehensive setup for a RAG (Retrieval Augmented Generation) agent. We can definitely structure this into a Python file for the agent, a FastAPI backend, a simple frontend, and Dockerize it.

Here's a plan and the code for each component:

**Project Structure:**

```
.
├── agent.py                 # Core LLM agent logic
├── main.py                  # FastAPI backend
├── frontend/
│   └── index.html           # Simple HTML frontend
├── storage/                 # This is where your LlamaIndex vector store is persisted
│   └── (index files...)
├── metadata.csv             # Needed if you were to rebuild index (optional for running pre-built)
├── extracted_texts.json     # Needed if you were to rebuild index (optional for running pre-built)
├── Dockerfile               # For building the backend Docker image
├── docker-compose.yml       # For running the application
└── requirements.txt         # Python dependencies
```

**Important Assumption:**
I'll assume that the vector index in the `storage/` directory is already built as per your notebook. The `agent.py` will focus on *loading* this pre-built index. If you need to build the index from scratch within Docker, the Dockerfile and agent script would need to include the data loading and indexing steps from your notebook, which can be time-consuming on first run.

### 1. `requirements.txt`

First, let's list the dependencies.

```text
fastapi
uvicorn[standard]
llama-index
llama-index-embeddings-huggingface
llama-index-llms-huggingface
bitsandbytes
torch
spacy
sentence-transformers
pandas
numpy
scikit-learn
nltk
matplotlib
python-multipart # For FastAPI form data if needed, good to have
```
*Note: `matplotlib` was in your notebook for plotting, it's not strictly necessary for the agent/backend but included for completeness if any part of your agent logic might use it internally. `bitsandbytes` is for quantization if your LLM setup requires it.*

You'll also need the spaCy model:
Run this once in your environment or ensure it's handled in the Dockerfile:
`python -m spacy download en_core_web_lg`

### 2. `agent.py`

This file will contain the logic to load your LLM, embedding model, vector index, and set up the chat engine.

import os
import torch
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.memory import ChatMemoryBuffer

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # or "meta-llama/Llama-2-7b-chat-hf"
PERSIST_DIR = "./storage" # Path to your persisted vector index

# System prompt for the agent
DEFAULT_SYSTEM_PROMPT = (
    "You are a medical chatbot, able to have normal interactions. "
    "You only answer based on the provided context. If the context does not contain "
    "the answer, state that the information is not available in the provided documents."
)

class RAGAgent:
    def __init__(self, persist_dir=PERSIST_DIR, model_name_embed=MODEL_NAME_EMBED,
                 model_name_llm=MODEL_NAME_LLM, device=DEVICE, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.persist_dir = persist_dir
        self.model_name_embed = model_name_embed
        self.model_name_llm = model_name_llm
        self.device = device
        self.system_prompt = system_prompt

        self.chat_engine = None
        self._setup_agent()

    def _setup_llm(self):
        print(f"Setting up LLM: {self.model_name_llm} on device: {self.device}")
        # Adjust quantization_config as needed, or remove if not using bitsandbytes
        # For TinyLlama, quantization might not be strictly necessary or supported in the same way as larger models.
        # If using a larger model like Llama-2-7b, bitsandbytes config would be more relevant.
        # Example for larger models (ensure bitsandbytes is correctly installed and compatible):
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16, # or torch.bfloat16
        #     bnb_4bit_use_double_quant=False,
        # )

        llm = HuggingFaceLLM(
            model_name=self.model_name_llm,
            tokenizer_name=self.model_name_llm,
            context_window=2048, # As per your notebook
            max_new_tokens=256,  # As per your notebook
            device_map="auto", # Automatically uses CUDA if available
            # model_kwargs={"quantization_config": bnb_config} # Enable if using bitsandbytes
            generate_kwargs={"temperature": 0.7, "do_sample": True}, # Adjusted from notebook for potentially more factual
        )
        Settings.llm = llm
        print("LLM setup complete.")
        return llm

    def _load_index(self):
        if not os.path.exists(self.persist_dir) or not os.listdir(self.persist_dir):
            print(f"Error: Index directory '{self.persist_dir}' not found or is empty.")
            print("Please ensure the vector index is built and available in the specified directory.")
            raise FileNotFoundError(f"Index not found at {self.persist_dir}")

        print(f"Loading index from {self.persist_dir}...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name_embed, device=self.device
        )
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully.")
        return index

    def _setup_chat_engine(self, index):
        print("Setting up chat engine...")
        # ChatMemoryBuffer is used to provide conversation history to the LLM.
        # token_limit defines how much of the recent conversation to remember.
        memory = ChatMemoryBuffer.from_defaults(token_limit=32000) # As per your notebook

        chat_engine = index.as_chat_engine(
            chat_mode="context", # Uses context from retrieved documents
            memory=memory,
            system_prompt=self.system_prompt,
            # similarity_top_k=5 # You can adjust how many documents are retrieved
        )
        print("Chat engine setup complete.")
        return chat_engine

    def _setup_agent(self):
        print(f"Initializing RAG Agent with device: {self.device}")
        self._setup_llm() # Sets Settings.llm
        index = self._load_index() # Sets Settings.embed_model
        self.chat_engine = self._setup_chat_engine(index)
        print("RAG Agent is ready.")

    def query(self, user_query: str) -> str:
        if not self.chat_engine:
            return "Error: Chat engine is not initialized."

        print(f"Received query: {user_query}")
        # For a stateless API, we might want to reset memory before each query,
        # or manage sessions if conversational memory is desired across API calls.
        # The notebook's interactive chat calls chat_engine.reset() after each response.
        # If you want each API call to be independent:
        self.chat_engine.reset()

        response = self.chat_engine.chat(user_query)
        # If you prefer streaming for the API (e.g., for Server-Sent Events):
        # response_stream = self.chat_engine.stream_chat(user_query)
        # full_response = ""
        # for token in response_stream.response_gen:
        #     full_response += token
        # return full_response

        print(f"Agent response: {response.response}")
        return response.response

# Example usage (for testing agent.py directly)
if __name__ == "__main__":
    print("Testing RAGAgent...")
    try:
        agent = RAGAgent()

        # Test query 1
        query1 = "Does smoking increase the risk of hospitalization for COVID-19 patients?"
        print(f"\nUser: {query1}")
        answer1 = agent.query(query1)
        print(f"Agent: {answer1}")

        # Test query 2 (potentially out of context)
        query2 = "What is the capital of France?"
        print(f"\nUser: {query2}")
        answer2 = agent.query(query2)
        print(f"Agent: {answer2}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

### 3. `main.py` (FastAPI Backend)

This file will create the API endpoints.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import agent as rag_agent_module # Assuming your agent file is named agent.py
import os

app = FastAPI(
    title="RAG Agent API",
    description="API for interacting with a Retrieval Augmented Generation agent.",
    version="0.1.0"
)

# --- Agent Initialization ---
# Ensure PERSIST_DIR is correct relative to where main.py is run, or use absolute paths.
# If Dockerizing, this path will be inside the container.
PERSIST_DIR_FASTAPI = "./storage" # Should match agent.py or be configured
if not os.path.exists(PERSIST_DIR_FASTAPI) or not os.listdir(PERSIST_DIR_FASTAPI):
    print(f"WARNING: Index directory '{PERSIST_DIR_FASTAPI}' not found or is empty during FastAPI startup.")
    print("The agent might fail to initialize. Ensure the 'storage' directory is correctly populated and accessible.")
    # You could choose to raise an error here to prevent the app from starting without the index
    # raise RuntimeError(f"Index directory '{PERSIST_DIR_FASTAPI}' is missing or empty. FastAPI app cannot start.")
    agent_instance = None # Set to None if index is missing
else:
    try:
        agent_instance = rag_agent_module.RAGAgent(persist_dir=PERSIST_DIR_FASTAPI)
    except Exception as e:
        print(f"Error initializing RAGAgent: {e}")
        agent_instance = None # Set to None if initialization fails


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_info: str = "Retrieved from knowledge base" # Placeholder for future source tracking

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

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Agent API. Use the /ask endpoint to submit questions."}

# To run this app: uvicorn main:app --reload

### 4. `frontend/index.html`

A very basic HTML frontend to interact with the API.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Agent Chat</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        input[type="text"] { width: calc(100% - 80px); padding: 10px; margin-right: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .response { margin-top: 20px; padding: 10px; background-color: #e9e9e9; border-radius: 4px; white-space: pre-wrap; }
        .loader { display: none; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ask the Medical RAG Agent</h1>
        <div>
            <input type="text" id="questionInput" placeholder="Type your question here...">
            <button onclick="askQuestion()">Ask</button>
        </div>
        <div class="loader" id="loader">Loading...</div>
        <div class="response" id="responseArea">
            Agent's answer will appear here.
        </div>
    </div>

    <script>
        async function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const responseArea = document.getElementById('responseArea');
            const loader = document.getElementById('loader');
            const question = questionInput.value;

            if (!question.trim()) {
                responseArea.textContent = 'Please enter a question.';
                return;
            }

            responseArea.textContent = '';
            loader.style.display = 'block';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question }),
                });

                loader.style.display = 'none';

                if (!response.ok) {
                    const errorData = await response.json();
                    responseArea.textContent = `Error: ${response.status} - ${errorData.detail || 'Failed to get answer'}`;
                    return;
                }

                const data = await response.json();
                responseArea.textContent = data.answer;
            } catch (error) {
                loader.style.display = 'none';
                responseArea.textContent = 'Failed to connect to the API: ' + error.message;
                console.error('Error:', error);
            }
        }
    </script>
</body>
</html>
```

### 5. `Dockerfile`

This will define the image for your backend application.

```dockerfile
# Start with a base Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any, e.g., for bitsandbytes or other libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Consider pre-downloading models here if they are large and static,
# or use a volume for Hugging Face cache.
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variable for Hugging Face cache (optional, good practice)
# This helps persist downloads across container rebuilds if you mount a volume here.
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache/transformers
ENV HF_DATASETS_CACHE=/app/huggingface_cache/datasets
ENV SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache/sentence_transformers
RUN mkdir -p $HF_HOME/transformers $HF_HOME/datasets $HF_HOME/sentence_transformers

# Command to run the FastAPI application
# Ensure main.py is in the root of the /app directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6. `docker-compose.yml`

This file will define and run your application services.

```yaml
version: '3.8'

services:
  rag_app:
    build: . # Build the image from the Dockerfile in the current directory
    container_name: medical_rag_agent_service
    ports:
      - "8000:8000" # Map port 8000 on the host to port 8000 in the container
    volumes:
      # Mount the storage directory containing the vector index
      # Replace ./storage with the actual path to your storage directory on the host if different
      - ./storage:/app/storage:ro # Mount as read-only if the app only reads the index
      # Mount the frontend directory to serve static files (if FastAPI serves it, or for dev)
      # For a simple HTML file, it's easier to just test by opening index.html in browser
      # and ensuring it points to localhost:8000/ask
      # - ./frontend:/app/frontend 
      
      # Mount Hugging Face cache to persist downloaded models (optional but recommended)
      # Create a directory e.g., ./huggingface_cache_volume on your host machine first
      - ./huggingface_cache_volume:/app/huggingface_cache 
    environment:
      # Pass any necessary environment variables to your application
      # e.g., HF_TOKEN if you use private models
      - HF_HOME=/app/huggingface_cache # Ensure this matches Dockerfile
      - TRANSFORMERS_CACHE=/app/huggingface_cache/transformers
      - HF_DATASETS_CACHE=/app/huggingface_cache/datasets
      - SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache/sentence_transformers
    # deploy: # Uncomment and configure if you have a CUDA-enabled Docker setup
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1 # Request 1 GPU
    #           capabilities: [gpu]
    command: uvicorn main:app --host 0.0.0.0 --port 8000

volumes:
  huggingface_cache_volume: # Defines the named volume for HF cache persistence
```

**Before Running with Docker:**

1.  **Build the Vector Index:** Make sure your `storage/` directory is populated with the index files generated by your Jupyter notebook. The `agent.py` and `main.py` assume this directory and its contents exist.
2.  **Hugging Face Cache Volume (Optional but Recommended):** Create a directory on your host machine (e.g., `mkdir ./huggingface_cache_volume`) to be used by the `docker-compose.yml` for persisting downloaded models. This saves time and bandwidth on subsequent runs.
3.  **GPU Access in Docker (If applicable):** If you intend to use a GPU within Docker, ensure your Docker setup supports NVIDIA GPUs (e.g., NVIDIA Container Toolkit installed). The `deploy` section in `docker-compose.yml` is commented out but shows how you might request GPU resources. The `device_map="auto"` in `agent.py` should automatically use the GPU if available to the container.

**How to Run:**

1.  **Place all files** in their respective locations as per the project structure.
2.  **Ensure your `storage/` directory is populated.**
3.  **Open a terminal** in the root directory of the project.
4.  **Build and run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    If you want to run in detached mode (in the background):
    ```bash
    docker-compose up --build -d
    ```
5.  **Access the API:**
    *   The API will be available at `http://localhost:8000`.
    *   You can see the root message at `http://localhost:8000/`.
    *   You can send POST requests to `http://localhost:8000/ask`.
6.  **Access the Frontend:**
    *   Open the `frontend/index.html` file directly in your web browser. It will make requests to `http://localhost:8000/ask`.

This setup provides a solid foundation. You can expand upon it by adding more sophisticated error handling, logging, user authentication, session management for conversational context, and a more advanced frontend. Remember that running LLMs, especially larger ones, can be resource-intensive.
```

You can serve your `frontend/index.html` (and other static files like CSS or JavaScript if you add them later) directly from your FastAPI application. This way, users can access your frontend by navigating to an endpoint like `http://localhost:8000/` in their browser.

Here's how you can modify your `main.py` to achieve this:

### Updated `main.py`

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from fastapi.responses import HTMLResponse # Import HTMLResponse
from pydantic import BaseModel
import agent as rag_agent_module
import os

app = FastAPI(
    title="RAG Agent API",
    description="API for interacting with a Retrieval Augmented Generation agent and serving its frontend.",
    version="0.1.0"
)

# --- Agent Initialization ---
PERSIST_DIR_FASTAPI = "./storage"
agent_instance = None
try:
    if os.path.exists(PERSIST_DIR_FASTAPI) and os.listdir(PERSIST_DIR_FASTAPI):
        agent_instance = rag_agent_module.RAGAgent(persist_dir=PERSIST_DIR_FASTAPI)
    else:
        print(f"WARNING: Index directory '{PERSIST_DIR_FASTAPI}' not found or is empty during FastAPI startup.")
        print("The agent might fail to initialize. Ensure the 'storage' directory is correctly populated and accessible.")
except Exception as e:
    print(f"Error initializing RAGAgent: {e}")
    # agent_instance remains None


# --- Mount static files (for CSS, JS, images if you add them) ---
# This will serve files from the 'frontend' directory under the path '/static'
# For example, if you have frontend/style.css, it would be accessible at /static/style.css
# app.mount("/static", StaticFiles(directory="frontend"), name="static")

# --- API Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_info: str = "Retrieved from knowledge base"

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    if agent_instance is None:
        if not os.path.exists(PERSIST_DIR_FASTAPI) or not os.listdir(PERSIST_DIR_FASTAPI):
            print("Startup: RAG Agent could not be initialized because the index is missing or empty.")
        else:
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

# --- Serve Frontend ---
# Ensure your 'frontend' directory is at the same level as main.py
# or adjust the path accordingly.
# The Dockerfile's `COPY . .` will copy the frontend directory into /app/frontend

# Option 1: Serve index.html directly at the root path
@app.get("/", response_class=HTMLResponse)
async def read_index():
    frontend_dir = "frontend" # Relative to where main.py is
    index_html_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_html_path):
        return HTMLResponse(content="<html><body><h1>Frontend not found</h1><p>Ensure index.html is in the 'frontend' directory.</p></body></html>", status_code=404)
    with open(index_html_path) as f:
        return HTMLResponse(content=f.read())

# Option 2 (Alternative): If you want to serve all files from 'frontend' directly
# without a /static prefix, and have index.html as the default for the root.
# This is more common for Single Page Applications (SPAs).
# Remove or comment out the @app.get("/") above if using this.
# And ensure your index.html <script> tags etc., don't assume a /static path unless you structure it that way.
#
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend_root")
#
# If you use this app.mount("/", ..., html=True), FastAPI will automatically serve
# 'index.html' from the 'frontend' directory when someone accesses '/'.
# Your `frontend/index.html` script tag for `/ask` will still work as it's an absolute path.

# To run this app: uvicorn main:app --reload

**Explanation of Changes:**

1.  **`from fastapi.responses import HTMLResponse`**: This is imported to allow FastAPI to return HTML content.
2.  **`@app.get("/", response_class=HTMLResponse)`**:
    *   This creates a new GET endpoint at the root path (`/`).
    *   `response_class=HTMLResponse` tells FastAPI to treat the return value as HTML.
3.  **`async def read_index():`**:
    *   This function defines what happens when a user goes to `http://localhost:8000/`.
    *   `frontend_dir = "frontend"`: Specifies the directory where your `index.html` is located. This path is relative to where `main.py` is executed. Inside the Docker container, if your `Dockerfile` has `COPY . .`, and your `frontend` directory is at the root of your project, then it will be at `/app/frontend`.
    *   `index_html_path = os.path.join(frontend_dir, "index.html")`: Constructs the full path to `index.html`.
    *   It checks if the file exists and returns a 404 if not.
    *   `with open(index_html_path) as f: return HTMLResponse(content=f.read())`: Reads the content of `index.html` and returns it as an HTML response.

**Alternative using `StaticFiles` for the root (more common for SPAs):**

If you prefer a more standard way to serve a directory of static files (including `index.html` as the default for `/`), you can use `app.mount` with `html=True`:

# In main.py, instead of the @app.get("/") route handler:

# from fastapi.staticfiles import StaticFiles # Ensure this is imported

# Mount the 'frontend' directory to be served at the root path '/'
# The `html=True` argument tells StaticFiles to serve 'index.html' for directory requests (like '/')
# Ensure this mount is placed AFTER your API routes (like /ask) if there's any path overlap,
# or ensure no overlap. For root mount, it's usually fine.
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend_static_root")
If you use this `app.mount("/", ...)` method, you would remove the `@app.get("/")` route handler that manually reads `index.html`. FastAPI will handle serving `frontend/index.html` when you go to `/`.

**Docker Considerations:**

*   Your current `Dockerfile` with `COPY . .` will copy the `frontend` directory (if it's in your project root alongside `main.py`, `agent.py`, etc.) into the `/app` directory in the container. So, inside the container, your `index.html` will be at `/app/frontend/index.html`. The path `frontend` used in `main.py` will correctly resolve to `/app/frontend` because the working directory in the container is `/app`.
*   The `docker-compose.yml` doesn't strictly need changes for this if the `Dockerfile` handles copying the `frontend` directory. The volume mount for `frontend` in `docker-compose.yml` (`- ./frontend:/app/frontend`) is useful for development as it reflects changes to your frontend code immediately without rebuilding the image. For production, copying via `Dockerfile` is standard.

**To Test:**

1.  Make the changes to `main.py`.
2.  Rebuild and restart your Docker containers:
    ```bash
    docker-compose up --build -d
    ```
3.  Now, open your web browser and navigate to `http://localhost:8000/`. You should see your `index.html` page served by FastAPI. The "Ask" button should still work as before, making requests to the `/ask` endpoint.
