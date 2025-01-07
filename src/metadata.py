import json
import time
import logging
from transformers import GPT2Tokenizer

from src.config import get_documents_table, get_force_rebuild, ollama_query, QUERY_MODEL, PROMPT_OPTIONS, CONTEXT_WINDOW
from src.classes import MetaInfo, create_GovDoc, create_MetaInfo, get_id_from_filename

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
documents_table = get_documents_table()

def run_prompt(prompt, label="generic", format: dict[str, any] = None):
  start_time = time.time()
  # Ensure the prompt fits within the context window
  response_tokens = 150  # Consider response tokens to avoid exceeding the context window
  max_prompt_tokens = CONTEXT_WINDOW - response_tokens
  # Tokenize the prompt with truncation to fit within the token limit
  tokens = tokenizer.encode(prompt, truncation=True, max_length=max_prompt_tokens)
  prompt = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
  try:
    response = ollama_query.generate(model=QUERY_MODEL,
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
  metadata_prompt = f"Please extract the following information from a document: 1.) title, 2.) summary, 3.) level_of_government, 4.) responsible_province, 5.) responsible_city, 6.) authors, 7.) editors 8.) publisher, 9.) publish_date, 10.) publisher_location, 11.) copyright_year, 12.) ISSN, 13.) ISBN, 14.) languages. If the exact title of the document is obvious in the text, then use that, alternatively the title should be your most releveant suggestion for the document and also be less than 8 words. The summary should be concise but still representative of the content of the text and also less than 50 words. Level of government is one of three options: 'federal', 'provincial', or 'municipal'. If the level of government is federal, the responsible province should be Ontario. Federal documents are Ottawa's responsibility. And provincial documents are the responsibility of the capital city of the responsible_province. Municipal documents are the responsibility of that city. The publish date should be converted to yyyy-mm-dd format. If found, write the ISBN number in this format: X-XXXX-XXXX-X. Detected languages should be one or both of these options: 'en', 'fr'. Only include a language if a significant portion of the text is in that language. You should output the information as JSON. Here follows the available document text:\n\n{text}"
  metadata = run_prompt(metadata_prompt, "metadata", format)
  return metadata


def clean_metadata_json(metadata):
  # Handle None/Null values
  for key, value in metadata.items():
    if metadata.get(key) in (None, "", "null"):
      metadata[key] = ""
    # Ensure that all number fields are strings
    value = metadata.get(key)
    if isinstance(value, int):
      metadata[key] = str(value)

  # Ensure that these fields are lists
  for key in ["authors", "editors", "languages"]:
    value = metadata.get(key)
    if not isinstance(value, list):
      if value in (None, "", "null"):
        metadata[key] = []  # Replace empty or null-like values with an empty list
      else:
        metadata[key] = [value]  # Convert single string values to a list
  return metadata


def extract_metadata(text:str, filename:str):
  doc_id = get_id_from_filename(filename)
  # Check if the file exists in the database
  existing_record = documents_table.to_pandas().query(f"doc_id == '{doc_id}'")
  if not existing_record.empty:
    existing_title = existing_record["title"].values[0]
    if existing_title and not get_force_rebuild():
      logging.info(f"Skipping metadata generation for {doc_id} (already exists)")
      return

  metadata_json_string = get_metadata(text)
  metadata = json.loads(metadata_json_string)
  metadata = clean_metadata_json(metadata)  # Handle None/Null values
  govdoc = create_GovDoc(create_MetaInfo(metadata), doc_id, filename)
  try:
    documents_table.merge_insert("filename").when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute([govdoc.model_dump()])
  except Exception as e:
    print(f"Error merging metadata: {e}")
