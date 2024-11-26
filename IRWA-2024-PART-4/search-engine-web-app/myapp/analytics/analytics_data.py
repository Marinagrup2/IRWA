import json
import random

from datetime import datetime
class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    def __init__(self):
        # Inicializar variables de instancia
        self.fact_clicks = {}  # Diccionario para guardar clics
        self.fact_queries = []  # Lista para guardar consultas
        self.clicked_docs = []  # Lista para guardar documentos clicados
        self.sessions = []  # Lista para guardar datos de sesiones


    def save_query_terms(self, terms: str) -> int:
        """
        Save query terms and generate a unique search ID.
        """
        search_id = random.randint(0, 100000)
        query_data = {
            "query": terms,
            "timestamp": datetime.now().isoformat(),
            "search_id": search_id
        }
        self.fact_queries.append(query_data)
        return search_id

    def save_click_data(self, doc_id: int, search_id: int):
        """
        Save document click data and update click statistics.
        """
        click_data = {
            "doc_id": doc_id,
            "search_id": search_id,
            "timestamp": datetime.now().isoformat()
        }
        self.clicked_docs.append(click_data)

        # Update fact_clicks dictionary
        if doc_id in self.fact_clicks:
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1

    def save_session(self, session_id: int, ip: str, user_agent: str, start_time: str, end_time: str = None):
        """
        Save session details.
        """
        session_data = {
            "session_id": session_id,
            "ip": ip,
            "user_agent": user_agent,
            "start_time": start_time,
            "end_time": end_time
        }
        self.sessions.append(session_data)
    
    def end_session(self, session_id: int, end_time: str):
        """
        Mark the end of a session.
        """
        for session in self.sessions:
            if session["session_id"] == session_id:
                session["end_time"] = end_time
                break



class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
