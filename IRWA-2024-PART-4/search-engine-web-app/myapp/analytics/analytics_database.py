import sqlite3
from datetime import datetime
import httpagentparser  # Asegúrate de tener este módulo instalado

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

    def save_session(self, session_id, ip, user_agent, start_time):
        """
        Save session data to the database (stored in the analytics table).
        """
        query = """
        INSERT INTO analytics (session_id, ip_address, browser, operating_system, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """
        # Extraer información del user_agent
        browser_info = httpagentparser.detect(user_agent)
        browser = browser_info.get('browser', {}).get('name', 'Unknown')
        operating_system = browser_info.get('os', {}).get('name', 'Unknown')

        self.execute_query(query, (session_id, ip, browser, operating_system, start_time))

    def save_query(self, session_id, query_text, timestamp):
        """
        Save a search query to the database (stored in the analytics table).
        """
        query = """
        INSERT INTO analytics (session_id, query, timestamp)
        VALUES (?, ?, ?)
        """
        self.execute_query(query, (session_id, query_text, timestamp))

    def save_click(self, session_id, doc_id, title, description, timestamp):
        """
        Save a document click to the database (stored in the analytics table).
        """
        query = """
        INSERT INTO analytics (session_id, doc_id, title, description, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """
        self.execute_query(query, (session_id, doc_id, title, description, timestamp))


    def end_session(self, session_id, end_time):
        """
        Mark the end of a session in the database (stored in the analytics table).
        """
        query = """
        UPDATE analytics
        SET timestamp = ?
        WHERE session_id = ?
        """
        self.execute_query(query, (end_time, session_id))