from pydantic import BaseModel,ConfigDict
from enum import Enum

class FileType(Enum):

    PDF:str=".pdf"
    TXT:str=".txt"
    DOCX:str=".docx"
    MD:str=".md"

class Valid_File(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    file_type:FileType

class Valid_Response(BaseModel):
    answer:str
    page_no:list[int]