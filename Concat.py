import requests
import logging
import sqlite3
import json
import hashlib
import time

def create_cache_table():
    """Create a cache table if it doesn't exist."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            url_hash TEXT PRIMARY KEY,
            response TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_call_count (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_cached_response(api_url):
    """Retrieve cached response if available."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    url_hash = hashlib.md5(api_url.encode()).hexdigest()
    cursor.execute("SELECT response FROM api_cache WHERE url_hash = ?", (url_hash,))
    row = cursor.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def cache_response(api_url, response):
    """Cache API response."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    url_hash = hashlib.md5(api_url.encode()).hexdigest()
    cursor.execute("REPLACE INTO api_cache (url_hash, response) VALUES (?, ?)", (url_hash, json.dumps(response)))
    conn.commit()
    conn.close()

def log_api_call():
    """Log an API call to the database."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO api_call_count DEFAULT VALUES")
    conn.commit()
    conn.close()

def get_api_call_count():
    """Retrieve the total number of API calls made."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM api_call_count")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_last_api_call_timestamp():
    """Retrieve the timestamp of the last API call."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM api_call_count ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "No API calls recorded."

def clear_cache():
    """Clear the API cache."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM api_cache")
    conn.commit()
    conn.close()
    logging.info("Cache cleared successfully.")

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
                log_api_call()
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

def load_from_file(filename="output.json"):
    """Load processed data from a JSON file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        logging.info(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
        return None

def get_cache_size():
    """Retrieve the number of cached entries."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM api_cache")
    count = cursor.fetchone()[0]
    conn.close()
    logging.info(f"Cache contains {count} entries.")
    return count

def export_cache_to_file(filename="cache_export.json"):
    """Export cache data to a JSON file."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT url_hash, response FROM api_cache")
    data = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
    conn.close()
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Cache exported to {filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_cache_table()
    url = "https://api.example.com/data"
    raw_data = fetch_data(url)
    processed_data = process_data(raw_data)
    save_to_file(processed_data)
    loaded_data = load_from_file()
    api_call_count = get_api_call_count()
    last_api_timestamp = get_last_api_call_timestamp()
    cache_size = get_cache_size()
    export_cache_to_file()
    logging.info(f"Total API calls made: {api_call_count}")
    logging.info(f"Last API call timestamp: {last_api_timestamp}")
    logging.info(f"Cache size: {cache_size}")
    clear_cache()
