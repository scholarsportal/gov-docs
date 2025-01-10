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
  response_tokens = 200  # Consider response tokens to avoid exceeding the context window
  max_prompt_tokens = CONTEXT_WINDOW - response_tokens
  # Tokenize the prompt with truncation to fit within the token limit
  tokens = tokenizer.encode(prompt, truncation=True, max_length=max_prompt_tokens)
  prompt = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
  try:
    response = ollama_query.generate(model=QUERY_MODEL,
                                     prompt=prompt,
                                     stream=False,
                                     options=PROMPT_OPTIONS,
                                     format="json")  # json.dumps(format)
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
  metadata_prompt = f"Please extract the following information from a document: 1.) title, 2.) summary, 3.) level_of_government, 4.) responsible_province, 5.) responsible_city, 6.) authors, 7.) editors 8.) publisher, 9.) publish_date, 10.) publisher_location, 11.) copyright_year, 12.) ISSN, 13.) ISBN, 14.) languages. If the exact title of the document is obvious in the text, then use that, alternatively the title should be your most releveant suggestion for the document and also be less than 8 words. The summary should be concise but still representative of the content of the text and also less than 50 words. Level of government is one of three options: 'federal', 'provincial', or 'municipal'. If the level of government is federal, the responsible province should be Ontario. Federal documents are Ottawa's responsibility. And provincial documents are the responsibility of the capital city of the responsible_province. Municipal documents are the responsibility of that city. The authors and editors lists should only contain strings of the respective names of authors and editors. The author can also be a person who signed an introductory letter at the beginning of the document. The publish date should be converted to yyyy-mm-dd format. If found, write the ISBN number in this format: X-XXXX-XXXX-X. Detected languages should be one or both of these options: 'en', 'fr'. Only include a language if a significant portion of the text is in that language. You should output the information as JSON. Here follows the available document text:\n\n{text}"
  metadata = run_prompt(metadata_prompt, "metadata", format)
  return metadata


def get_catergory_keywords(text):
  format = {
      "type": "object",
      "properties": {
          "keywords": {
              "type": "array",
              "items": {
                  "type": "string"
              }
          },
          "category": {
              "type": "string"
          }
      }
  }
  category_prompt = f"Categorize a document into one of the following categories (specific definitions provided here to aid picking the best category):\n* Financial and Operational Reports (Reports from ministries or agencies detailing their activities and finances. Includes annual reports, budgets, expenditure estimates, public accounts, statements, and fiscal summaries)\n* Research and Analysis (In-depth examinations of specific topics, including research reports, discussion papers, and documents that summarize public feedback and consultations)\n* News and Media (Documents designed for public communication, including bulletins, notices, news releases, backgrounders, newsletters, and speeches by government officials)\n* Policies and Directives (Documents that outline rules, regulations, and best practices, including policies, directives, manuals, guidelines, standards, and codes)\n* Strategic and Operational Plans (Documents that outline goals, objectives, and plans for the future, including strategic plans, mandate letters, and ministerial objectives)\n* Promotional and Educational Material (Documents designed to educate or promote government initiatives or public awareness, including brochures, pamphlets, flyers, educational content, and informational guides)\n\nOutput the results in JSON format.\nDocument text follows:\n\n{text}"
  prompt = f"Create metadata fields `keywords` and `category` for a document.\n\nExtract the 5 best keywords from a document to aid in indexing and searchability.\nKeywords are words or short phrases that are less than 3 words that help to categorize and index the document for easier retrieval and searchability in databases and search engines. They enable researchers and readers to quickly identify the relevant subject matter and scope of the document.\nEach keyword entry should not have more than two words.\nDon't include keywords represented by the document title.\n\nCategorize the document into one of the following categories:\n* Financial and Operational Reports\n* Research and Analysis\n* News and Media\n* Policies and Directives\n* Strategic and Operational Plans\n* Promotional and Educational Material\n\nOutput the results in JSON format.\nDocument text follows:\n\n{text}"
  category = run_prompt(prompt, "category", format)
  return category


def clean_metadata_json(metadata):
  # Handle None/Null values
  for key, value in metadata.items():
    if metadata.get(key) in (None, "", "null", "unknown"):
      metadata[key] = ""
    # Ensure that all number fields are strings
    value = metadata.get(key)
    if isinstance(value, int):
      metadata[key] = str(value)

  # Ensure that these fields are lists
  for key in ["authors", "editors", "languages", "keywords"]:
    value = metadata.get(key)
    if not isinstance(value, list):
      if value in (None, "", "null"):
        metadata[key] = []  # Replace empty or null-like values with an empty list
      else:
        metadata[key] = [value]  # Convert single string values to a list
  return metadata


def extract_metadata(text: str, filename: str):
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
  try:
    govdoc = create_GovDoc(create_MetaInfo(metadata), doc_id, filename)
  except Exception as e:
    logging.error(f"Error mapping metadata: {e}")
    print(metadata)
    return
  cat_json_string = get_catergory_keywords(text)
  cat = json.loads(cat_json_string)
  cat = clean_metadata_json(cat)  # Handle None/Null values
  govdoc.keywords = cat.get("keywords")
  govdoc.category = cat.get("category")
  try:
    documents_table.merge_insert("filename").when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute([govdoc.model_dump()])
  except Exception as e:
    logging.error(f"Error merging metadata: {e}")
