import os
from ollama import Client
import lancedb
from dotenv import load_dotenv
from src.classes import new_GovDoc, new_Embedding

# Load environment variables from .env file
load_dotenv()

# Constants
OLLAMA_EMBED_URL = "https://ollama.scholarsportal.info/ollama"
# OLLAMA_EMBED_URL = "https://openwebui.zacanbot.com/ollama"
EMBEDDING_MODEL = "snowflake-arctic-embed2:latest"
EMBED_API_KEY = os.getenv('EMBED_API_KEY')
if not EMBED_API_KEY:
  raise ValueError("EMBED_API_KEY not found in environment variables")
EMBED_HEADERS = {"Authorization": f"Bearer {EMBED_API_KEY}"}
OLLAMA_QUERY_URL = "https://ollama.scholarsportal.info/ollama"
# OLLAMA_QUERY_URL = "https://openwebui.zacanbot.com/ollama"
QUERY_MODEL = "llama3.2-vision:11b-instruct-q8_0"
# QUERY_MODEL = "qwen2.5-coder:14b-instruct-q8_0"
# QUERY_MODEL = "phi4:14b-q8_0"
# QUERY_MODEL = "qwen2.5-coder:14b-instruct-q6_K"
# QUERY_MODEL = "dolphin3:8b-llama3.1-q8_0"
# QUERY_MODEL = "tulu3:8b-q8_0"
QUERY_API_KEY = os.getenv('QUERY_API_KEY')
if not QUERY_API_KEY:
  raise ValueError("QUERY_API_KEY not found in environment variables")
QUERY_HEADERS = {"Authorization": f"Bearer {QUERY_API_KEY}"}
CONTEXT_WINDOW = 4096
PROMPT_OPTIONS = {"temperature": 0.0, "num_ctx": CONTEXT_WINDOW}
VECTOR_DB_PATH = "./lance_db"
FORCE_REBUILD = False  # Set to True to rebuild the vector store from scratch
DEBUG = False

# Initialize Ollama clients
ollama_embed = Client(
    host=OLLAMA_EMBED_URL,
    headers=EMBED_HEADERS,
    timeout=30,
)
ollama_query = Client(
    host=OLLAMA_QUERY_URL,
    headers=QUERY_HEADERS,
    timeout=30,
)
# Initialize LanceDB
vector_db = lancedb.connect(VECTOR_DB_PATH)

def get_debug():
  return DEBUG

def get_force_rebuild():
  return FORCE_REBUILD

def set_parameters(debug: bool = False, force_rebuild: bool = False):
  global DEBUG, FORCE_REBUILD
  DEBUG = debug
  FORCE_REBUILD = force_rebuild

def get_documents_table():
  if "documents" not in vector_db.table_names():
    # Provide a sample record to define the schema
    sample_data = [new_GovDoc().model_dump()]
    documents_table = vector_db.create_table("documents", sample_data)
  else:
    documents_table = vector_db.open_table("documents")
  return documents_table

def get_embeddings_table():
  if "embeddings" not in vector_db.table_names():
    # Provide a sample record to define the schema
    sample_data = [new_Embedding().model_dump()]
    embeddings_table = vector_db.create_table("embeddings", sample_data)
  else:
    embeddings_table = vector_db.open_table("embeddings")
  return embeddings_table
