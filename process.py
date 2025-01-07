import os
import json
import logging
import numpy as np
from pathlib import Path

from src.config import set_parameters, get_documents_table
from src.embed import embed_document
from src.metadata import extract_metadata


def embed_documents(files):
  for file in files:
    logging.info(f"Embedding  {file.name}...")
    with file.open('r', encoding='utf-8') as f:
      text = f.read()
    embed_document(text, file.name)


def generate_metadata(files):
  for file in files:
    logging.info(f"Generating metadata for {file.name}...")
    with file.open('r', encoding='utf-8') as f:
      text = f.read()
    extract_metadata(text, file.name)


def export_metadata():
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
    export_metadata()


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
