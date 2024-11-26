import sqlite3
from datetime import datetime

class AnalyticsDatabase:
    """Handles SQLite database operations."""

    def __init__(self, db_path='analytics.db'):
        """
        Initialize the database object with the given path.
        """
        self.db_path = db_path

    def init_db(self):
        """
        Initialize the database with the schema.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Leer el archivo de esquema
            with open('schema.sql', 'r') as f:
                schema = f.read()
            # Ejecutar múltiples sentencias SQL usando executescript
            cursor.executescript(schema)
            conn.commit()

    def execute_query(self, query, params=None):
        """
        Execute a single query on the database.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)  # Ejecuta una sola sentencia con parámetros
            else:
                cursor.execute(query)  # Ejecuta una sola sentencia sin parámetros
            conn.commit()

    def save_session(self, session_id, ip, user_agent, start_time, end_time=None):
        """
        Save session data to the database.
        """
        query = """
        INSERT INTO session (session_id, ip, user_agent, start_time, end_time)
        VALUES (?, ?, ?, ?, ?)
        """
        self.execute_query(query, (session_id, ip, user_agent, start_time, end_time))

    def save_query(self, session_id, query_text, timestamp):
        """
        Save a search query to the database.
        """
        query = """
        INSERT INTO query (session_id, query, timestamp)
        VALUES (?, ?, ?)
        """
        self.execute_query(query, (session_id, query_text, timestamp))

    def save_click(self, session_id, doc_id, timestamp):
        """
        Save a document click to the database.
        """
        query = """
        INSERT INTO click (session_id, doc_id, timestamp)
        VALUES (?, ?, ?)
        """
        self.execute_query(query, (session_id, doc_id, timestamp))

    def end_session(self, session_id, end_time):
        """
        Mark the end of a session in the database.
        """
        query = """
        UPDATE session
        SET end_time = ?
        WHERE session_id = ?
        """
        self.execute_query(query, (end_time, session_id))
