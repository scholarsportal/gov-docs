from pydantic import BaseModel
from typing import List, Optional


class MetaInfo(BaseModel):
  title: Optional[str] = None
  level_of_government: Optional[str] = None
  responsible_province: Optional[str] = None
  responsible_city: Optional[str] = None
  authors: Optional[List[str]] = None
  editors: Optional[List[str]] = None
  publisher: Optional[str] = None
  publish_date: Optional[str] = None
  publisher_location: Optional[str] = None
  copyright_year: Optional[str] = None
  ISSN: Optional[str] = None
  ISBN: Optional[str] = None
  language: Optional[List[str]] = None
  summary: Optional[str] = None


def create_MetaInfo(json: dict) -> MetaInfo:
  return MetaInfo(**json)


def new_MetaInfo() -> MetaInfo:
  return MetaInfo(title="",
                  level_of_government="",
                  responsible_province="",
                  responsible_city="",
                  authors=[""],
                  editors=[""],
                  publisher="",
                  publish_date="",
                  publisher_location="",
                  copyright_year="",
                  ISSN="",
                  ISBN="",
                  language=[""],
                  summary="")


class GovDoc(MetaInfo):
  doc_id: str
  filename: str


def get_id_from_filename(filename: str) -> str:
  # id is the last part of the file path without the filename extension
  return filename.split("/")[-1].split(".")[0]


def create_GovDoc(metadata: MetaInfo, filename: str) -> GovDoc:
  id = get_id_from_filename(filename)
  return GovDoc(**metadata.model_dump(), doc_id=id, filename=filename)


def new_GovDoc() -> GovDoc:
  return create_GovDoc(new_MetaInfo(), "sample.txt")


class Embedding(BaseModel):
  doc_id: str
  chunk_id: int
  content: str
  embedding: list[float]


def create_Embedding(json: dict) -> Embedding:
  return Embedding(**json)


def new_Embedding() -> Embedding:
  return Embedding(doc_id="", chunk_id=0, content="", embedding=[0.0])
