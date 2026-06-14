import json
import os
from datetime import datetime

STORAGE_DIR = "chat_storage"
SESSIONS_FILE = f"{STORAGE_DIR}/sessions.json"

os.makedirs(STORAGE_DIR, exist_ok=True)

def load_all_sessions():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_session(session_id, pdf_name, chat_history):
    sessions = load_all_sessions()
    sessions[session_id] = {
        "pdf_name": pdf_name,
        "chat_history": chat_history,
        "last_updated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "message_count": len([m for m in chat_history if m["role"] == "user"])
    }
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def delete_session(session_id):
    sessions = load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def load_session(session_id):
    sessions = load_all_sessions()
    return sessions.get(session_id, None)

def save_analytics(event_type, data={}):
    analytics_file = f"{STORAGE_DIR}/analytics.json"
    analytics = {}
    if os.path.exists(analytics_file):
        try:
            with open(analytics_file, "r") as f:
                analytics = json.load(f)
        except:
            analytics = {}

    if "events" not in analytics:
        analytics["events"] = []
    if "totals" not in analytics:
        analytics["totals"] = {
            "pdfs_processed": 0,
            "questions_asked": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "sessions": 0
        }

    analytics["events"].append({
        "type": event_type,
        "data": data,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
    })

    # Update totals
    if event_type == "pdf_processed":
        analytics["totals"]["pdfs_processed"] += 1
        analytics["totals"]["sessions"] += 1
    elif event_type == "question_asked":
        analytics["totals"]["questions_asked"] += 1
        conf = data.get("confidence", "Medium")
        if conf == "High":
            analytics["totals"]["high_confidence"] += 1
        elif conf == "Low":
            analytics["totals"]["low_confidence"] += 1
        else:
            analytics["totals"]["medium_confidence"] += 1

    # Keep only last 500 events
    analytics["events"] = analytics["events"][-500:]

    with open(analytics_file, "w") as f:
        json.dump(analytics, f, indent=2)

def load_analytics():
    analytics_file = f"{STORAGE_DIR}/analytics.json"
    if not os.path.exists(analytics_file):
        return {"totals": {}, "events": []}
    try:
        with open(analytics_file, "r") as f:
            return json.load(f)
    except:
        return {"totals": {}, "events": []}