import os
import json
import requests
from pathlib import Path
import lancedb
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Constants
OLLAMA_API_URL = "https://openwebui.zacanbot.com/ollama"
EMBEDDING_MODEL = "snowflake-arctic-embed2:latest"
RAG_MODEL = "llama3.2-vision:11b-instruct-q8_0"
AUTH_HEADER = f"Bearer {os.getenv('API_KEY')}"
OLLAMA_HEADERS = {
    'Authorization': AUTH_HEADER,
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}
VECTOR_DB_PATH = "./lance_db"
FORCE_REBUILD = False  # Set to True to rebuild the vector store from scratch

# Initialize LanceDB
vector_db = lancedb.connect(VECTOR_DB_PATH)
if "documents" not in vector_db.table_names():
  # Provide a sample record to define the schema
  sample_data = [{"filename": "example.txt", "embedding": [0.0] * 384}]
  embeddings_table = vector_db.create_table("documents", sample_data)
else:
  embeddings_table = vector_db.open_table("documents")


def get_embedding(text):
  data = {"model": EMBEDDING_MODEL, "input": text}
  response = requests.post(f"{OLLAMA_API_URL}/api/embed", json=data, headers=OLLAMA_HEADERS)
  response.raise_for_status()
  return response.json()['embeddings']


def get_title(text):
  title_prompt = f"Please provide an appropriate title for my document. Do not explain your answer. Only respond with your best title. Here follows my document:\n{text}"
  data_title = {"model": RAG_MODEL, "prompt": title_prompt, "stream": "false"}
  response_title = requests.post(f"{OLLAMA_API_URL}/api/generate",
                                 json=data_title,
                                 headers=OLLAMA_HEADERS)
  response_title.raise_for_status()
  title = response_title.json()['response'].strip()
  return title


def get_summary(text):
  summary_prompt = f"Please summarize this text in 30 words or less:\n{text}"
  data_summary = {"model": RAG_MODEL, "prompt": summary_prompt, "stream": "false"}
  response_summary = requests.post(f"{OLLAMA_API_URL}/api/generate",
                                   json=data_summary,
                                   headers=OLLAMA_HEADERS)
  response_summary.raise_for_status()
  summary = response_summary.json()['response'].strip()
  return summary


def embed_documents(files):
  for file in files:
    filename = file.name
    print(f"Processing {filename}...")
    # Check if the embedding already exists in the table
    existing = embeddings_table.to_pandas().query(f"filename == '{filename}'")

    # Check if embedding already exists unless --force is specified
    if not FORCE_REBUILD and not existing.empty:
      print(f"Skipping embedding for {filename} (already exists)")
      continue

    with file.open('r', encoding='utf-8') as f:
      text = f.read()
    embedding = get_embedding(text)
    # Store the embedding
    flat_embedding = [float(val) for sublist in embedding for val in sublist]
    embeddings_table.merge_insert("filename").when_matched_update_all() \
      .when_not_matched_insert_all() \
      .execute([{"filename": filename, "embedding": flat_embedding}])


def generate_metadata(files):
  metadata = {}
  for file in files:
    filename = file.name
    print(f"Generating summary and title for {filename}...")
    with file.open('r', encoding='utf-8') as f:
      text = f.read()
    title = get_title(text)
    summary = get_summary(text)
    metadata[filename] = {"title": title, "summary": summary}
  # Save to metadata.json
  with open("metadata.json", 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
  print(f"Metadata saved to metadata.json")
  print(metadata)


def main(input_path):
  if not os.path.exists(input_path):
    print("The specified folder or file does not exist.")
  else:
    input_path = Path(input_path)
    if input_path.is_file():
      files = [input_path]
    else:
      files = list(input_path.rglob("*.txt"))

    embed_documents(files)
    generate_metadata(files)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Embed documents and generate summaries/titles.")
  parser.add_argument("input",
                      help="Path to the folder containing .txt files or a single .txt file")
  parser.add_argument("--force",
                      action="store_true",
                      help="Force embedding even if embeddings already exist")
  args = parser.parse_args()
  FORCE_REBUILD = args.force
  main(args.input)
