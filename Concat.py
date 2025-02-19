import requests
import logging
import sqlite3
import json
import hashlib

def create_cache_table():
    """Create a cache table if it doesn't exist."""
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            url_hash TEXT PRIMARY KEY,
            response TEXT
        )
    ""
    )
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

def fetch_data(api_url):
    """Fetch data from the given API URL and return the JSON response."""
    logging.info(f"Fetching data from {api_url}")
    cached_data = get_cached_response(api_url)
    if cached_data:
        logging.info("Using cached data")
        return cached_data
    response = requests.get(api_url)
    if response.status_code == 200:
        logging.info("Data fetched successfully")
        json_data = response.json()
        cache_response(api_url, json_data)
        return json_data
    else:
        logging.error(f"Failed to fetch data. Status code: {response.status_code}")
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
