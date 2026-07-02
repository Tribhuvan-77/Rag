from fastapi import FastAPI,File,UploadFile,HTTPException

import os
import shutil
from pathlib import Path
from uuid import uuid4
from schema import Valid_File
import rag
app=FastAPI()

class TypeFile():
    file_type:str

    def __init__(self,file_type):
        self.file_type=file_type


@app.post("/upload")
async def post_upload_file(query:str,file:UploadFile=File(...)):
    typefile=TypeFile(Path(file.filename).suffix)
    
    try:
     valid_filetype=Valid_File.model_validate(typefile)
    except Exception as e:
       raise HTTPException(status_code=400,detail=str(e))
    
    
    
    UPLOAD_DIR="upload"

    os.makedirs(UPLOAD_DIR,exist_ok=True)

    unique_filename=f"{uuid4()}{valid_filetype.file_type}"

    file_path=os.path.join(UPLOAD_DIR,unique_filename)
    with open(file_path,"wb") as buffer:
       shutil.copyfileobj(file.file,buffer)
    

    result=rag.Load_File(file_path,valid_filetype.file_type.value,query)

    return result
    
    
    
    
    


    

