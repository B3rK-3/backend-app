import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn =  psycopg2.connect(
        database="postgres",
        user=os.environ["user"],
        password=os.environ["pass"],
        host=os.environ["host"],
        port=os.environ["port"],
    )

