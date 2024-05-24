import sqlite3

def initialize_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Create the subjects table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        year INTEGER NOT NULL,
        total_classes INTEGER NOT NULL
    )
    """)

    # Create the attendance table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        subject_id INTEGER NOT NULL,
        entrytime TEXT,
        exittime TEXT,
        date DATE,
        FOREIGN KEY(subject_id) REFERENCES subjects(id),
        UNIQUE(name, subject_id, date)
    )
    """)

    conn.commit()
    conn.close()

if __name__ == '__main__':
    initialize_database()
