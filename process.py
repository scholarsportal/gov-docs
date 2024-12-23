import os
import json
import logging
from pathlib import Path
import lancedb
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from transformers import GPT2Tokenizer
# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import time
from pydantic import BaseModel
from ollama import Client

# Constants
OLLAMA_API_URL = "https://openwebui.zacanbot.com/ollama"
EMBEDDING_MODEL = "snowflake-arctic-embed2:latest"
RAG_MODEL = "llama3.2-vision-11b-q8_0:latest"
API_KEY = os.getenv('API_KEY')
if not API_KEY:
  raise ValueError("API_KEY environment variable not set")
OLLAMA_HEADERS = {"Authorization": f"Bearer {API_KEY}"}
CONTEXT_WINDOW = 4096
PROMPT_OPTIONS = {"temperature": 0.1, "num_ctx": CONTEXT_WINDOW}
VECTOR_DB_PATH = "./lance_db"
FORCE_REBUILD = False  # Set to True to rebuild the vector store from scratch
DEBUG = False

# Initialize Ollama client
ollama = Client(
    host=OLLAMA_API_URL,
    headers=OLLAMA_HEADERS,
)
# Initialize LanceDB
vector_db = lancedb.connect(VECTOR_DB_PATH)
if "documents" not in vector_db.table_names():
  # Provide a sample record to define the schema
  sample_data = [{"filename": "example.txt", "embedding": [0.0] * 384}]
  embeddings_table = vector_db.create_table("documents", sample_data)
else:
  embeddings_table = vector_db.open_table("documents")


def init():
  # Initialize logging
  logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='%(message)s')


def get_embedding(text):
  try:
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response['embeddings']
  except Exception as e:
    logging.error(f"Error getting embedding: {e}")
    return None


def run_prompt(prompt, label="generic", format: dict[str, any] = None):
  start_time = time.time()
  # Ensure the prompt fits within the context window
  response_token_limit = 150  # Limit response tokens to avoid exceeding the context window
  max_prompt_tokens = CONTEXT_WINDOW - response_token_limit
  # Tokenize the prompt with truncation to fit within the token limit
  tokens = tokenizer.encode(prompt, truncation=True, max_length=max_prompt_tokens)
  prompt = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
  try:
    response = ollama.generate(model=RAG_MODEL,
                               prompt=prompt,
                               stream=False,
                               options=PROMPT_OPTIONS,
                               format="json")  # change to format when ollama client adds support
    answer = response['response'].strip()
    end_time = time.time()  # End timer
    logging.debug(answer)
    logging.info(f"** {label} result in {end_time - start_time:.1f}s")
    return answer
  except Exception as e:
    logging.error(f"Error running prompt: {e}")
    return None


def get_metadata(text):

  class MetaInfo(BaseModel):
    title: str
    summary: str
    level_of_government: str
    responsible_province: str
    responsible_city: str
    authors: list[str]
    editors: list[str]
    publisher: str
    publish_date: str
    publisher_location: str
    copyright_year: int
    ISSN: str
    ISBN: str
    language: list[str]

  format = MetaInfo.model_json_schema()
  metadata_prompt = f"Please extract the following information from a document: 1.) title, 2.) summary, 3.) level_of_government, 4.) responsible_province, 5.) responsible_city, 6.) authors, 7.) editors 8.) publisher, 9.) publish_date, 10.) publisher_location, 11.) copyright_year, 12.) ISSN, 13.) ISBN, 14.) language. If the exact title of the document is obvious in the text, then use that, alternatively the title should be your most releveant suggestion for the document and also be less than 8 words. The summary should be concise but still representative of the content of the text and also less than 50 words. Level of government is one of three options: 'federal', 'provincial', or 'municipal'. If the level of government is federal, the responsible province should be Ontario. Federal documents are Ottawa's responsibility. And provincial documents are the responsibility of the capital city of the responsible_province. Municipal documents are the responsibility of that city. The publish date should be converted to yyyy-mm-dd format. Language should be one or both of these options: 'en', 'fr'. You should output the information as JSON. Here follows the available document text:\n\n{text}"
  metadata = run_prompt(metadata_prompt, "metadata", format)
  return metadata


def get_title(text):
  title_prompt = f"Please infer an appropriate title for a piece of text. You should only output the title. Here follows the text:\n\n{text}"
  title = run_prompt(title_prompt, "title")
  # remove leading and trailing quotes from title if they exist
  if (title[0] == '"'):
    title = title[1:-1]
  return title


def get_summary(text):
  summary_prompt = f"Please summarize a piece of text in 50 words or less and do not explain your answer. You should only output the summary. Here follows the text:\n\n{text}"
  summary = run_prompt(summary_prompt, "summary")
  return summary


def embed_documents(files):
  for file in files:
    filename = file.name
    text = ""
    # Check if the embedding already exists in the table
    existing = embeddings_table.to_pandas().query(f"filename == '{filename}'")
    if not FORCE_REBUILD and not existing.empty:
      logging.info(f"Skipping embedding for {filename} (already exists)")
      continue

    logging.info(f"Embedding  {filename}...")
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
    text = ""
    logging.info(f"Processing {filename}...")
    with file.open('r', encoding='utf-8') as f:
      text = f.read()
      # title = get_title(text)
      # summary = get_summary(text)
      # metadata[filename] = {"title": title, "summary": summary}
      json_text = get_metadata(text)
      metadata[filename] = json.loads(json_text)
  # Save to metadata.json
  with open("metadata.json", 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
  logging.info(f"Metadata saved to metadata.json")


def main(input_path):
  if not os.path.exists(input_path):
    logging.info("The specified folder or file does not exist.")
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
  parser.add_argument("--debug", action="store_true", help="Executes the script in debug mode")
  args = parser.parse_args()
  FORCE_REBUILD = args.force
  DEBUG = args.debug
  init()
  main(args.input)
