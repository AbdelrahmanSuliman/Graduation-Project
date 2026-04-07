import sqlite3

def init_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            glass_id INTEGER NOT NULL,
            detected_face_shape_id INTEGER NOT NULL,
            -- Biometric Features used for the MLP branch
            cheek_jaw_ratio REAL,
            face_hw_ratio REAL,
            midface_ratio REAL,
            liked BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_user_feedback(glass_id: int, shape_id: int, features: list, liked: bool):
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (glass_id, detected_face_shape_id, cheek_jaw_ratio, face_hw_ratio, midface_ratio, liked)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (glass_id, shape_id, features[0], features[1], features[2], liked))
    conn.commit()
    conn.close()