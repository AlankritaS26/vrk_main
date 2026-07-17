"""
session.py - RNS Digital Receptionist (Session Manager)
All session data stored in PostgreSQL only. No local files.
"""

import os, time, requests
from dotenv import load_dotenv
from database import get_db_connection

load_dotenv()

BACKEND = os.getenv('BACKEND_URL', 'http://127.0.0.1:8001')

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
        conn = get_db_connection()
        if not conn: return
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO interactions (session_id, input_text, response_text) VALUES (%s, %s, %s)',
            (session_id, text if speaker == 'user' else '', text if speaker == 'kiosk' else '')
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f'[DB] save message error: {e}')

def end_session_in_db(session_id):
    try:
        conn = get_db_connection()
        if not conn: return
        cur = conn.cursor()
        cur.execute(
            'UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE session_id = %s',
            (session_id,)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f'[SESSION] Closed in PostgreSQL -> {session_id}')
    except Exception as e:
        print(f'[DB] end session error: {e}')

def run():
    print('[SESSION MANAGER] Running. All data stored in PostgreSQL only.')

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
            message_index = 0

        # Active session - save new messages to PostgreSQL
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