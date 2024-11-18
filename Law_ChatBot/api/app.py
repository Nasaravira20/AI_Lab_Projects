from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('Generator.py'))))

from Retriever.Generator import process

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    query: str

@app.post("/legal-advice")
async def get_legal_advice(query_model: QueryModel):
    try:
        # Call the process function
        result = process(query_model.query)
        if result:
            return {"result": result}
        else:
            raise HTTPException(status_code=404, detail="No result found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
