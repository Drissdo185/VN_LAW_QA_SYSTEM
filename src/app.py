from fastapi import FastAPI, HTTPException
from rag.indexing import Indexing

app = FastAPI()
index_manager = Indexing(path="./chroma_db", chunk_size=200, chunk_overlap=10)

@app.post("/index")
async def create_index():
    try:
        index_manager.embed_and_index()
        return {"message": "Indexing process completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query")
async def query_index(query: str):
    try:
        index_manager.load_index()
        results = index_manager.retrieve(query)
        if not results:
            raise HTTPException(status_code=404, detail="No documents found for the query.")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
