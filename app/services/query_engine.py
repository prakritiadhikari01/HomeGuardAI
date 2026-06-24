# app/services/query_engine.py
from app.services.embedderai import get_embedding
from app.services.vector_storeai import search_events

def answer_query(question: str) -> dict:
    embedding = get_embedding(question)
    events = search_events(embedding, n_results=5)

    if not events:
        return {
            "answer": "No matching events found for your query.",
            "events": []
        }

    lines = [f"Found {len(events)} matching event(s):\n"]
    for i, e in enumerate(events, 1):
        label = e["person_label"] if e["person_label"] != "unknown" else "Unknown person"
        lines.append(
            f"{i}. [{e['timestamp']}] at {e['location']} (Camera: {e['camera_id']})\n"
            f"   Person: {label} ({e['person_type']})\n"
            f"   Clothing: {e['clothing_description']}\n"
            f"   Action: {e['action_description']}\n"
        )

    return {
        "answer": "\n".join(lines),
        "events": events
    }