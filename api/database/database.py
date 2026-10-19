import sqlite3
import bcrypt
from src.v1.config import DB_FILE

#initialising database
def init_db():
    """Run this once when your application starts up."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                hashed_password TEXT NOT NULL
            )
        """)

# for registering user and pass
def register_user(username, password):
    
    # prepare the credentials
    password_bytes = password.encode('utf-8')
    
    salt = bcrypt.gensalt()
    
    hashed = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (username, hashed)
            )
        return True  # automatically committed on successful exit of 'with' block
    
    except sqlite3.IntegrityError:
        return False

# check if the user already registered
def verify_user(username, password):
    
    with sqlite3.connect(DB_FILE) as conn:
        
        cursor = conn.cursor()
        
        cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
        
        row = cursor.fetchone()
        
    if row:
        stored_hash = row[0].encode('utf-8')
        
        password_bytes = password.encode('utf-8')
        
        return bcrypt.checkpw(password_bytes, stored_hash)
    
    # if the user doesn't exists return false
    return False