import requests
from bs4 import BeautifulSoup
import csv
import json
import sqlite3
import concurrent.futures
import time
from textblob import TextBlob

# Start time tracking
start_time = time.time()

# Database setup
conn = sqlite3.connect("quotes.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        author TEXT,
        author_url TEXT,
        birth_date TEXT,
        birth_place TEXT,
        tags TEXT,
        scrape_time TEXT,
        sentiment TEXT,
        length INTEGER,
        word_count INTEGER,
        popularity_score INTEGER,  -- New Feature: Popularity Score
        source TEXT  -- New Feature: Verified Source
    )
""")
conn.commit()

def get_sentiment(text):
    """Determine sentiment of a quote."""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def get_quote_length(text):
    """Calculate the length of the quote."""
    return len(text)

def get_word_count(text):
    """Calculate the word count of the quote."""
    return len(text.split())

def get_popularity_score(text):
    """Fetch estimated quote popularity using Bing Search API."""
    API_KEY = "your_bing_api_key"
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": f'"{text}"', "count": 1}

    try:
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("webPages", {}).get("totalEstimatedMatches", 0)
    except Exception as e:
        log_error(f"Error fetching popularity score: {e}")
        return 0  # Default to 0 if API fails

def get_verified_source(text):
    """Fetch the verified source of a quote using Wikiquote or another source."""
    # This is a placeholder implementation; you can improve it using an API
    sources = {
        "The only thing we have to fear is fear itself.": "Franklin D. Roosevelt, 1933 Inaugural Address",
        "I think, therefore I am.": "René Descartes, Discourse on the Method"
    }
    return sources.get(text, "Unknown")

# Error logging setup
def log_error(message):
    with open("scraper_errors.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

base_url = "http://quotes.toscrape.com"
page_num = 1
quotes_list = []

while True:
    URL = f"{base_url}/page/{page_num}/"
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        log_error(f"Error fetching {URL}: {e}")
        break

    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")
    if not quotes:
        break  # Exit loop if no more quotes are found

    for quote in quotes:
        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()
        tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]
        author_url = base_url + quote.find("a")["href"] if quote.find("a") else "N/A"
        scrape_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        quote_data = {
            'text': text,
            'author': author,
            'author_url': author_url,
            'tags': tags,
            'scrape_time': scrape_time,
            'sentiment': get_sentiment(text),
            'length': get_quote_length(text),
            'word_count': get_word_count(text),
            'popularity_score': get_popularity_score(text),  # New Feature
            'source': get_verified_source(text)  # New Feature
        }

        quotes_list.append(quote_data)

    page_num += 1

# Insert data into SQLite
for quote in quotes_list:
    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source']))
    conn.commit()

conn.close()

# Saving data to files
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

# Save to CSV
with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

# Save quote count to a text file
with open("quote_count.txt", "w") as countfile:
    countfile.write(f"Total Quotes Scraped: {len(quotes_list)}\n")

# End time tracking and display execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Quotes saved to all_quotes.json, all_quotes.csv, quote_count.txt, and database.")
print(f"Total execution time: {execution_time:.2f} seconds")
