import os
import json
import logging
from pathlib import Path
import numpy as np
from transformers import GPT2Tokenizer

from src.config import get_documents_table, ollama, RAG_MODEL, PROMPT_OPTIONS, CONTEXT_WINDOW, set_parameters, get_force_rebuild
from src.classes import MetaInfo, create_GovDoc, create_MetaInfo, get_id_from_filename
from src.embed import embed_document

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import time


def run_prompt(prompt, label="generic", format: dict[str, any] = None):
  start_time = time.time()
  # Ensure the prompt fits within the context window
  response_tokens = 150  # Consider response tokens to avoid exceeding the context window
  max_prompt_tokens = CONTEXT_WINDOW - response_tokens
  # Tokenize the prompt with truncation to fit within the token limit
  tokens = tokenizer.encode(prompt, truncation=True, max_length=max_prompt_tokens)
  prompt = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
  try:
    response = ollama.generate(model=RAG_MODEL,
                               prompt=prompt,
                               stream=False,
                               options=PROMPT_OPTIONS,
                               format="json") # json.dumps(format)
    answer = response['response'].strip()
    end_time = time.time()  # End timer
    logging.debug(answer)
    logging.info(f"-> {label} result in {end_time - start_time:.1f}s")
    return answer
  except Exception as e:
    logging.error(f"Error running prompt: {e}")
    return None


def get_metadata(text):

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
    doc_id = get_id_from_filename(file.name)
    logging.info(f"Embedding  {doc_id}...")
    with file.open('r', encoding='utf-8') as f:
      text = f.read()
    embed_document(text, doc_id)


def clean_metadata(metadata):
  # Handle None/Null values
  for key, value in metadata.items():
    if metadata.get(key) in (None, "", "null"):
      metadata[key] = ""
    # Ensure that all number fields are strings
    value = metadata.get(key)
    if isinstance(value, int):
      metadata[key] = str(value)

  # Ensure that these fields are lists
  for key in ["authors", "editors", "language"]:
    value = metadata.get(key)
    # Ensure these fields are lists
    if not isinstance(value, list):
      if value in (None, "", "null"):
        metadata[key] = [""]  # Replace empty or null-like values with an empty list
      else:
        metadata[key] = [value]  # Convert single string values to a list
  return metadata


def generate_metadata(files):
  documents_table = get_documents_table()
  records = documents_table.to_pandas()

  for file in files:
    filename = file.name
    # Check if the file exists in the database
    existing_record = records.query(f"filename == '{filename}'")
    if not existing_record.empty:
      existing_title = existing_record["title"].values[0]
      if existing_title and not get_force_rebuild():
        logging.info(f"Skipping metadata generation for {filename} (already exists)")
        continue

    logging.info(f"Generating metadata for {filename}...")
    with file.open('r', encoding='utf-8') as f:
      text = f.read()

    # Generate metadata using the appropriate model
    metadata_json_string = get_metadata(text)
    metadata = json.loads(metadata_json_string)
    metadata = clean_metadata(metadata)  # Handle None/Null values
    govdoc = create_GovDoc(create_MetaInfo(metadata), filename)
    try:
      documents_table.merge_insert("filename").when_matched_update_all() \
          .when_not_matched_insert_all() \
          .execute([govdoc.model_dump()])
    except Exception as e:
      print(f"Error merging metadata: {e}")


def export_metadata_to_csv():
  try:
    # Convert the table to a pandas DataFrame
    data = get_documents_table().to_pandas()
    # Exclude the first record (sample data)
    data = data.iloc[1:]
    # Convert ndarray or other non-serializable objects to Python lists
    for column in data.columns:
      if data[column].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        data[column] = data[column].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)

    # Write the DataFrame to a CSV file
    data.to_csv("metadata.csv", index=False, encoding='utf-8')
    # Export to JSON
    with open("metadata.json", 'w', encoding='utf-8') as json_file:
      metadata = data.to_dict(orient="records")
      json.dump(metadata, json_file, ensure_ascii=False, indent=2)
    print(f"Metadata successfully exported to metadata.csv and metadata.json")
  except Exception as e:
    print(f"Error exporting metadata: {e}")


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
    export_metadata_to_csv()


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
  set_parameters(args.debug, args.force)
  logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='%(message)s')
  main(args.input)
