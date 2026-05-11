import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# 1. This looks at your .env file and grabs your API keys
load_dotenv()

# 2. This sets up the connection to Pinecone (your "Memory")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("semantic-cache")

# 3. This sets up the connection to OpenAI (your "Brain")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 4. This creates your Web API
app = FastAPI()

# This is a simple template for what the user sends us (a prompt)
class QueryRequest(BaseModel):
    prompt: str

    # Add this at the top with your other variables
stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "total_latency_saved_ms": 0
}

@app.get("/stats")
async def get_stats():
    # This is a 'Pro' move for your resume: showing you care about performance
    hit_rate = (stats["cache_hits"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0
    return {
        "cache_hit_rate": f"{hit_rate:.2f}%",
        "requests_handled": stats["total_requests"],
        "dollars_saved_estimate": f"${stats['cache_hits'] * 0.01:.4f}", # Assuming 1 cent per GPT-4 call
        "performance_boost": f"{stats['total_latency_saved_ms']}ms saved"
    }

@app.get("/")
def home():
    return {"message": "The Semantic Router is running. Go to /docs to test it!"}

@app.post("/query")
async def handle_query(request: QueryRequest):
    stats["total_requests"] += 1

    # 1. Check Cache first
    embedding_response = client.embeddings.create(
        input=request.prompt,
        model="text-embedding-3-small"
    )
    query_vector = embedding_response.data[0].embedding
    search_results = index.query(vector=query_vector, top_k=1, include_metadata=True)

    if search_results['matches'] and search_results['matches'][0]['score'] > 0.96:
        stats["cache_hits"] += 1
        stats["total_latency_saved_ms"] += 800 
        return {
            "source": "CACHE",
            "answer": search_results['matches'][0]['metadata']['answer']
        }

    # 2. ROUTER: Decide model based on prompt length or keywords
    # Simple logic: If prompt is short (< 15 chars) or contains simple words, use mini model.
    # Otherwise, use the "Big" model.
    if len(request.prompt) < 20:
        selected_model = "gpt-4o-mini"
        route_type = "Simple Route (Cheap)"
    else:
        selected_model = "gpt-4o"
        route_type = "Complex Route (High Quality)"

    # 3. Call the selected model
    llm_response = client.chat.completions.create(
        model=selected_model,
        messages=[{"role": "user", "content": request.prompt}]
    )
    answer = llm_response.choices[0].message.content

    # 4. Save to Cache
    index.upsert(vectors=[{
        "id": request.prompt[:50], 
        "values": query_vector, 
        "metadata": {"prompt": request.prompt, "answer": answer}
    }])

    return {
        "source": route_type,
        "model_used": selected_model,
        "answer": answer
    }