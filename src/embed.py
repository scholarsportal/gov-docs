import re
import logging

from src.classes import Embedding
from src.config import get_embeddings_table, ollama, EMBEDDING_MODEL, get_force_rebuild

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 500

embeddings_table = get_embeddings_table()


def clean_and_normalize_text(text):
  """
    Cleans noisy text by removing scan artifacts, excessive whitespace, and illegible characters.
    """
  # Remove artifacts like page numbers, extra symbols, and excessive whitespace
  text = re.sub(r'(Page \d+:?|[‘"“”\'`~!@#$%^&*_+=|\{\}\[\]<>/\\]+)', '', text)
  text = re.sub(r'\s*\n\s*', '\n', text)  # Normalize newlines
  text = re.sub(r'\n{2,}', '\n\n', text)  # Ensure double newlines separate paragraphs
  text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-ASCII characters
  return text.strip()


def chunk_text(text, min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE) -> list[str]:
  """
    Splits the document into chunks based on paragraphs and token/word limits,
    ensuring each chunk is at least `min_chunk_size` while staying under `max_chunk_size`.
    :return: A list of chunks.
    """
  # Clean and preprocess document
  text = clean_and_normalize_text(text)
  paragraphs = text.split('\n\n')  # Split by paragraphs (double newline)

  chunks = []
  current_chunk = []
  current_size = 0

  for paragraph in paragraphs:
    paragraph_size = len(paragraph.split())

    # If adding this paragraph exceeds max_chunk_size, finalize the current chunk
    if current_size + paragraph_size > max_chunk_size:
      if current_size >= min_chunk_size:
        chunks.append(" ".join(current_chunk).strip())
        current_chunk = []
        current_size = 0
      else:
        # Add paragraph even if it exceeds max_chunk_size to ensure minimum size
        current_chunk.append(paragraph)
        current_size += paragraph_size
        chunks.append(" ".join(current_chunk).strip())
        current_chunk = []
        current_size = 0
      continue

    # Add paragraph to the current chunk
    current_chunk.append(paragraph)
    current_size += paragraph_size

  # Add the last chunk if it meets the minimum size requirement
  if current_chunk and current_size >= min_chunk_size:
    chunks.append(" ".join(current_chunk).strip())

  # If the last chunk is too small, merge it with the previous chunk if possible
  elif chunks and current_chunk:
    last_chunk = chunks.pop()
    merged_chunk = f"{last_chunk} {' '.join(current_chunk)}".strip()
    chunks.append(merged_chunk)

  return chunks


# Alternative chunking with overlap
def chunk_overlap(text, max_chunk_length=200, overlap=20):
  result = []
  current_chunk_count = 0
  separator = ["\n", " "]
  _splits = re.split(f"({separator})", text)
  splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

  for i in range(len(splits)):
    if current_chunk_count != 0:
      chunk = "".join(splits[current_chunk_count - overlap:current_chunk_count + max_chunk_length])
    else:
      chunk = "".join(splits[0:max_chunk_length])
    if len(chunk) > 0:
      result.append("".join(chunk))
    current_chunk_count += max_chunk_length

  return result


def print_chunks(chunks):
  for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n{'-'*80}")


def get_embedding(text):
  try:
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response['embeddings']
  except Exception as e:
    logging.error(f"Error getting embedding: {e}")
    return None


def embed_document(text, doc_id):
  # Check if the embedding already exists in the table
  existing = embeddings_table.to_pandas().query(f"doc_id == '{doc_id}'")
  if not get_force_rebuild() and not existing.empty:
    logging.info(f"Skipping embedding for {doc_id} (already exists)")
    return
  if not existing.empty:
    # Delete existing embeddings for this document
    embeddings_table.delete(f"doc_id = '{doc_id}'")

  chunks = chunk_text(text)
  embeddings = []
  for chunk in chunks:
    embedding = get_embedding(chunk)
    if embedding is None:  # Handle failed embeddings
      logging.error(f"Failed to get embedding. Cancelling embedding for {doc_id}")
      return
    embeddings.append(embedding)

  # Save embeddings to LanceDB
  for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    flat_embedding = [float(val) for sublist in embedding for val in sublist]
    embed_model = Embedding(doc_id=doc_id, chunk_id=i, content=chunk, embedding=flat_embedding)
    # Save the embedding to the database
    embeddings_table.add([embed_model.model_dump()])
