from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from memory_system import MemorySystem

app = FastAPI(title="AML Text Memory System")
memory_store = MemorySystem()

# ---------- 定义AML要求的标准请求体 ----------
class AddRequest(BaseModel):
    user_id: str
    session_id: str
    content: str
    timestamp: Optional[str] = None

class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: Optional[int] = 5

class AddResponse(BaseModel):
    status: str
    memory_id: str

class SearchResponse(BaseModel):
    status: str
    memories: List[Dict[str, Any]]

# ---------- 健康检查（Smoke测试必看） ----------
@app.get("/")
def root():
    return {"status": "healthy", "message": "AML Memory System is running!"}

# ---------- Add 接口 ----------
@app.post("/add", response_model=AddResponse)
async def add_memory(req: AddRequest):
    try:
        mem_id = memory_store.add_memory(
            user_id=req.user_id,
            session_id=req.session_id,
            content=req.content,
            timestamp=req.timestamp
        )
        return AddResponse(status="success", memory_id=mem_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Search 接口 ----------
@app.post("/search", response_model=SearchResponse)
async def search_memory(req: SearchRequest):
    try:
        results = memory_store.search_memory(
            user_id=req.user_id,
            query=req.query,
            top_k=req.top_k
        )
        return SearchResponse(status="success", memories=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 本地运行测试（如果直接跑这个文件） ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)