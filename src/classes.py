from pydantic import BaseModel
import numpy as np

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

class GovDoc(MetaInfo):
  id: str
  filename: str
  embedding: np.ndarray
