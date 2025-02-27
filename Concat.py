import requests
import logging
import sqlite3
import json
import hashlib
import time
import os

DB_PATH = os.getenv("CACHE_DB_PATH", "cache.db")

def create_cache_table():
    """Create necessary cache tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            url_hash TEXT PRIMARY KEY,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_call_count (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            count INTEGER DEFAULT 1,
            total_response_time REAL DEFAULT 0,
            average_response_time REAL DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            status_code INTEGER,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_cache_status(status_type):
    """Log cache hit or miss."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO cache_stats (type) VALUES (?)", (status_type,))
    conn.commit()
    conn.close()

def log_request(api_url, status_code, response_time):
    """Log API request details."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO request_logs (url, status_code, response_time) 
        VALUES (?, ?, ?)
    """, (api_url, status_code, response_time))
    cursor.execute("""
        INSERT INTO api_call_count (url, count, total_response_time, average_response_time) 
        VALUES (?, 1, ?, ?) 
        ON CONFLICT(url) DO UPDATE 
        SET count = count + 1, 
            total_response_time = total_response_time + ?, 
            average_response_time = total_response_time / count
    """, (api_url, response_time, response_time, response_time))
    conn.commit()
    conn.close()

def log_error(error_message):
    """Log errors to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO error_logs (error_message) VALUES (?)", (error_message,))
    conn.commit()
    conn.close()

def get_cache_summary():
    """Retrieve cache statistics summary."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT type, COUNT(*) FROM cache_stats GROUP BY type")
    stats = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in stats}

def get_request_logs():
    """Retrieve API request logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM request_logs ORDER BY timestamp DESC LIMIT 10")
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_recent_requests(limit=10):
    """Retrieve the most recent API requests."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT url, status_code, response_time, timestamp FROM request_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_error_logs():
    """Retrieve error logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM error_logs ORDER BY timestamp DESC LIMIT 10")
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_most_requested_urls(limit=5):
    """Retrieve the most frequently requested API URLs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT url, count, average_response_time FROM api_call_count ORDER BY count DESC LIMIT ?", (limit,))
    urls = cursor.fetchall()
    conn.close()
    return urls

def clear_old_cache(expiry_time=86400):
    """Remove cache entries older than the expiry time."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM api_cache WHERE timestamp < datetime('now', '-{} seconds')".format(expiry_time))
    conn.commit()
    conn.close()

def clear_all_cache():
    """Remove all cache entries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM api_cache")
    conn.commit()
    conn.close()

def get_cached_response(api_url, cache_expiry=86400):
    """Retrieve cached response if available and not expired."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    url_hash = hashlib.md5(api_url.encode()).hexdigest()
    cursor.execute("SELECT response, timestamp FROM api_cache WHERE url_hash = ?", (url_hash,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        cached_time = time.mktime(time.strptime(row[1], "%Y-%m-%d %H:%M:%S"))
        if time.time() - cached_time < cache_expiry:
            log_cache_status("hit")
            return json.loads(row[0])
    log_cache_status("miss")
    return None

def cache_response(api_url, response):
    """Cache API response."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    url_hash = hashlib.md5(api_url.encode()).hexdigest()
    cursor.execute("REPLACE INTO api_cache (url_hash, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (url_hash, json.dumps(response)))
    conn.commit()
    conn.close()

def exponential_backoff(attempt, base_delay=1, max_delay=60):
    """Calculate exponential backoff time."""
    return min(base_delay * (2 ** attempt), max_delay)

def fetch_data(api_url, retries=3):
    """Fetch data from the given API URL with retry mechanism and return the JSON response."""
    logging.info(f"Fetching data from {api_url}")
    cached_data = get_cached_response(api_url)
    if cached_data:
        logging.info("Using cached data")
        return cached_data
    
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = requests.get(api_url, timeout=10)
            duration = time.time() - start_time
            logging.info(f"Request completed in {duration:.2f} seconds")
            log_request(api_url, response.status_code, duration)
            
            if response.status_code == 200:
                logging.info("Data fetched successfully")
                json_data = response.json()
                cache_response(api_url, json_data)
                return json_data
            else:
                logging.error(f"Failed to fetch data. Status code: {response.status_code}")
                return None
        except requests.RequestException as e:
            error_message = f"Attempt {attempt + 1} failed: {e}"
            logging.error(error_message)
            log_error(error_message)
            time.sleep(exponential_backoff(attempt))
    
    logging.error("Max retries reached. Unable to fetch data.")
    return None
