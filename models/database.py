import psycopg2
import os

DB_HOST = os.getenv("DB_HOST", "your-rds-endpoint")
DB_NAME = os.getenv("DB_NAME", "your-db-name")
DB_USER = os.getenv("DB_USER", "your-username")
DB_PASS = os.getenv("DB_PASS", "your-password")

def save_feedback(text, sentiment):
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (text, sentiment) VALUES (%s, %s)", (text, sentiment))
    conn.commit()
    cur.close()
    conn.close()
