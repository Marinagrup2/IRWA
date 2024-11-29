import csv
import os
from datetime import datetime
import httpagentparser

class AnalyticsCSV:
    """Handles CSV file operations for analytics data."""

    def __init__(self, file_path='analytics.csv'):
        """
        Initialize the CSV file object with the given path.
        """
        self.file_path = file_path

    def init_csv(self):
        """
        Initialize the CSV file with the header row if the file does not already exist.
        """
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the header row if the file doesn't exist
                writer.writerow(['session_id', 'ip_address', 'browser', 'operating_system', 'timestamp', 'query', 'doc_id', 'title', 'description'])

    def append_to_csv(self, data):
        """
        Append a new row of data to the CSV file.
        """
        with open(self.file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def save_session(self, session_id, ip, user_agent, start_time):
        """
        Save session data to the CSV file.
        """
        # Extract information from the user_agent
        agent = httpagentparser.detect(user_agent)

        # Extract browser and OS information, falling back to 'Unknown' if not found
        browser = agent['browser']['name']
        operating_system = agent['os']['name']

        timestamp = start_time
        data = [session_id, ip, browser, operating_system, timestamp, '', '', '', '']
        self.append_to_csv(data)

    def save_query(self, session_id, query_text, timestamp):
        """
        Save a search query to the CSV file.
        """
        data = [session_id, '', '', '', timestamp, query_text, '', '', '']
        self.append_to_csv(data)

    def save_click(self, session_id, doc_id, title, description, timestamp):
        """
        Save a document click to the CSV file.
        """
        data = [session_id, '', '', '', timestamp, '', doc_id, title, description]
        self.append_to_csv(data)

'''
    def end_session(self, session_id, end_time):
        """
        Mark the end of a session in the CSV file.
        We won't update the CSV directly but can mark end of session using `timestamp`.
        """
        # Read all rows from the CSV, modify the relevant session, and write back to CSV
        rows = []
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)

        for row in rows:
            if row[0] == str(session_id):  # Find the session by session_id
                row[4] = end_time  # Update the timestamp (end time)

        # Write the updated data back to the CSV
        with open(self.file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
            '''