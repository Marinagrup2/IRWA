import sqlite3

def count_clicks(db_path='analytics.db'):
    """
    Count the number of clicks in the database.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Contar los clics en la tabla click
        cursor.execute("SELECT COUNT(*) FROM click")
        total_clicks = cursor.fetchone()[0]
        print(f"Total clicks: {total_clicks}")

if __name__ == "__main__":
    count_clicks()
