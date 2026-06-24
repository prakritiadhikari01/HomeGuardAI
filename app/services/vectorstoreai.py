# app/services/vector_storeai.py
import chromadb

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _client.get_or_create_collection("home_events")
    return _collection

def store_event(event_id: str, text: str, embedding: list, metadata: dict):
    col = get_collection()
    # ChromaDB only accepts str/int/float/bool in metadata
    clean = {k: str(v) for k, v in metadata.items()}
    col.add(
        ids=[event_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[clean]
    )
    print(f"[VectorStore] Stored event {event_id}")

def search_events(query_embedding: list, n_results: int = 5) -> list:
    col = get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    output = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            output.append({
                "description": doc.strip(),
                "timestamp": meta.get("timestamp", "unknown"),
                "camera_id": meta.get("camera_id", "unknown"),
                "location": meta.get("location", "unknown"),
                "clothing_description": meta.get("clothing_description", ""),
                "action_description": meta.get("action_description", ""),
                "person_type": meta.get("person_type", "UNKNOWN"),
                "person_label": meta.get("person_label", "unknown"),
            })
    return output