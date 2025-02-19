import requests
import logging

def fetch_data(api_url):
    """Fetch data from the given API URL and return the JSON response."""
    logging.info(f"Fetching data from {api_url}")
    response = requests.get(api_url)
    if response.status_code == 200:
        logging.info("Data fetched successfully")
        return response.json()
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
    import json
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Data saved to {filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    url = "https://api.example.com/data"
    raw_data = fetch_data(url)
    processed_data = process_data(raw_data)
    save_to_file(processed_data)
