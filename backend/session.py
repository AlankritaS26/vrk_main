"""
session.py - RNS Digital Receptionist (Session Manager)
All session data stored in MongoDB locally. No local files.
"""

import os
import time
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

load_dotenv()

# Configs
BACKEND = os.getenv('BACKEND_URL', 'http://127.0.0.1:8001')
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/rnsit_db')

# Initialize MongoDB Client
# (Extracts the database name from the URI, defaulting to 'rnsit_db')
client = MongoClient(MONGO_URI)
db_name = MONGO_URI.split('/')[-1] if '/' in MONGO_URI.replace('://', '') else 'rnsit_db'
db = client[db_name]

def get_current_session():
    try:
        r = requests.get(f'{BACKEND}/session/current', timeout=2)
        data = r.json()
        return data if data.get('active') else None
    except Exception:
        return None

def get_new_messages(session_id, after_index):
    try:
        r = requests.get(
            f'{BACKEND}/session/messages/{session_id}',
            params={'after': after_index},
            timeout=2
        )
        return r.json().get('messages', [])
    except Exception:
        return []

def save_message_to_db(session_id, speaker, text):
    try:
        # Targeting the 'interactions' collection
        db.interactions.insert_one({
            "session_id": session_id,
            "input_text": text if speaker == 'user' else '',
            "response_text": text if speaker == 'kiosk' else '',
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f'[MongoDB] save message error: {e}')

def end_session_in_db(session_id):
    try:
        # Update the specific session document in 'sessions' collection
        db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "is_active": False,
                "ended_at": datetime.utcnow().isoformat()
            }},
            upsert=True # Creates it if it doesn't exist yet
        )
        print(f'[SESSION] Closed in MongoDB -> {session_id}')
    except Exception as e:
        print(f'[MongoDB] end session error: {e}')

def run():
    print('[SESSION MANAGER] Running. All data stored in MongoDB locally.')

    current_session_id = None
    message_index      = 0

    while True:
        session = get_current_session()

        # New session started
        if session and session.get('session_id') != current_session_id:
            current_session_id = session['session_id']
            user_name          = session.get('user_name', 'Guest')
            is_returning       = session.get('is_returning', False)
            print(f'[SESSION] Started -> ID: {current_session_id}')
            print(f'[SESSION] Visitor: {user_name} ({"returning" if is_returning else "new"})')
            
            # Formally log the session start into MongoDB
            try:
                db.sessions.update_one(
                    {"session_id": current_session_id},
                    {"$set": {
                        "session_id": current_session_id,
                        "user_name": user_name,
                        "is_active": True,
                        "is_returning": is_returning,
                        "started_at": datetime.utcnow().isoformat()
                    }},
                    upsert=True
                )
            except Exception as e:
                print(f'[MongoDB] session init error: {e}')
                
            message_index = 0

        # Active session - save new messages to MongoDB
        elif session and current_session_id:
            new_msgs = get_new_messages(current_session_id, message_index)
            for msg in new_msgs:
                speaker = msg.get('speaker', 'user')
                text    = msg.get('text', '')
                print(f'  [{speaker.upper()}] {text}')
                save_message_to_db(current_session_id, speaker, text)
                message_index += 1

        # Session ended
        elif not session and current_session_id:
            print(f'[SESSION] Ended -> {current_session_id}')
            end_session_in_db(current_session_id)
            current_session_id = None
            message_index      = 0
            print('[SESSION MANAGER] Waiting for next visitor...\n')

        time.sleep(1)

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print('\n[SESSION MANAGER] Stopped.')