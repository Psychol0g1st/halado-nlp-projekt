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
