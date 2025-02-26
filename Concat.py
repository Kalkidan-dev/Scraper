import requests
import logging
import sqlite3
import json
import hashlib
import time
import os

def create_cache_table():
    """Create necessary cache tables if they don't exist."""
    conn = sqlite3.connect("cache.db")
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
    conn.commit()
    conn.close()

def log_cache_status(status_type):
    """Log cache hit or miss."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO cache_stats (type) VALUES (?)", (status_type,))
    conn.commit()
    conn.close()

def clear_old_cache(expiry_time=86400):
    """Remove cache entries older than the expiry time."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM api_cache WHERE timestamp < datetime('now', '-{} seconds')".format(expiry_time))
    conn.commit()
    conn.close()

def get_cached_response(api_url, cache_expiry=86400):
    """Retrieve cached response if available and not expired."""
    conn = sqlite3.connect("cache.db")
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
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    url_hash = hashlib.md5(api_url.encode()).hexdigest()
    cursor.execute("REPLACE INTO api_cache (url_hash, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (url_hash, json.dumps(response)))
    conn.commit()
    conn.close()

def fetch_data(api_url, retries=3, backoff_factor=1):
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
            
            if response.status_code == 200:
                logging.info("Data fetched successfully")
                json_data = response.json()
                cache_response(api_url, json_data)
                return json_data
            else:
                logging.error(f"Failed to fetch data. Status code: {response.status_code}")
                return None
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    
    logging.error("Max retries reached. Unable to fetch data.")
    return None

def process_data(data):
    """Process the fetched data and return meaningful results."""
    if data:
        return {key: value for key, value in data.items() if value}
    return {}

def save_to_file(data, filename="output.json"):
    """Save processed data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_cache_table()
    url = "https://api.example.com/data"
    raw_data = fetch_data(url)
    processed_data = process_data(raw_data)
    save_to_file(processed_data)
