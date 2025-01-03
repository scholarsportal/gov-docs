from pydantic import BaseModel

class MetaInfo(BaseModel):
  title: str
  level_of_government: str
  responsible_province: str
  responsible_city: str
  authors: list[str]
  editors: list[str]
  publisher: str
  publish_date: str
  publisher_location: str
  copyright_year: str
  ISSN: str
  ISBN: str
  language: list[str]
  summary: str

def create_MetaInfo(json: dict) -> MetaInfo:
  return MetaInfo(**json)

def new_MetaInfo() -> MetaInfo:
  return MetaInfo(
    title="",
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
    summary=""
  )

class GovDoc(MetaInfo):
  id: str
  filename: str
  embedding: list[float]

def create_GovDoc(metadata: MetaInfo, filename: str, embedding: list[float]) -> GovDoc:
  # id is the last part of the file path without the filename extension
  id = filename.split("/")[-1].split(".")[0]
  return GovDoc(**metadata.model_dump(), id=id, filename=filename, embedding=embedding)

def new_GovDoc() -> GovDoc:
  return create_GovDoc(new_MetaInfo(), "sample.txt", [0.0] * 3)
