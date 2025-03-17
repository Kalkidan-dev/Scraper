import requests
from bs4 import BeautifulSoup
import csv
import json
import sqlite3
import concurrent.futures
import time
from textblob import TextBlob
import uuid
from langdetect import detect  # Language detection
import gender_guesser.detector as gender  # Gender prediction library

# Start time tracking
start_time = time.time()

# Database setup
conn = sqlite3.connect("quotes.db")
cursor = conn.cursor()

# Create table if not exists (with new "occupation" column)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quote_id TEXT,  -- New Feature: Unique Quote ID
        text TEXT,
        author TEXT,
        author_url TEXT,
        birth_date TEXT,
        birth_place TEXT,
        tags TEXT,
        scrape_time TEXT,
        sentiment TEXT,  -- New Column
        length INTEGER,  -- New Feature: Quote Length
        word_count INTEGER,  -- New Feature: Word Count
        source TEXT,  -- New Feature: Quote Source
        language TEXT,  -- New Feature: Quote Language
        gender TEXT,  -- New Feature: Author's Gender
        nationality TEXT,  -- New Feature: Author's Nationality
        occupation TEXT  -- New Feature: Author's Occupation
    )
""")
conn.commit()

# Initialize gender detector
d = gender.Detector()

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


def detect_language(text):
    """Detect the language of the quote."""
    try:
        return detect(text)
    except:
        return "unknown"  # Return unknown if detection fails


def get_author_gender(author_name):
    """Predict the gender of the author."""
    return d.get_gender(author_name.split()[0])  # We predict based on the first name


def get_author_nationality(author_name):
    """Predict the nationality of the author based on their first name."""
    api_url = f"https://api.genderize.io?name={author_name.split()[0]}"
    try:
        response = requests.get(api_url)
        data = response.json()
        nationality = data.get("country_id", "Unknown")
        return nationality
    except Exception as e:
        return "Unknown"  # In case the API fails or returns no data


def get_author_occupation(author_url):
    """Fetch the author's occupation."""
    try:
        author_response = requests.get(author_url, timeout=10)
        author_response.raise_for_status()
        author_soup = BeautifulSoup(author_response.text, "html.parser")
        # Trying to scrape occupation from author's page (this is an example, depends on the structure of the site)
        occupation = author_soup.find("div", class_="author-occupation")  # Placeholder class
        if occupation:
            return occupation.get_text().strip()
        else:
            return "Unknown"  # If no occupation is found
    except requests.exceptions.RequestException as e:
        return "Unknown"  # In case of error


# Error logging setup
def log_error(message):
    with open("scraper_errors.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

base_url = "http://quotes.toscrape.com"
page_num = 1
quotes_list = []

def fetch_author_details(author_url):
    """Fetch author birth details in a separate thread."""
    try:
        author_response = requests.get(author_url, timeout=10)
        author_response.raise_for_status()
        author_soup = BeautifulSoup(author_response.text, "html.parser")
        birth_date = author_soup.find("span", class_="author-born-date").get_text() if author_soup.find("span", class_="author-born-date") else "N/A"
        birth_place = author_soup.find("span", class_="author-born-location").get_text() if author_soup.find("span", class_="author-born-location") else "N/A"
        return birth_date, birth_place
    except requests.exceptions.RequestException as e:
        log_error(f"Error fetching author details from {author_url}: {e}")
        return "N/A", "N/A"

while True:
    URL = f"{base_url}/page/{page_num}/"
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching {URL}: {e}"
        print(error_message)
        log_error(error_message)
        break

    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")
    if not quotes:
        break  # Exit loop if no more quotes are found

    author_futures = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for quote in quotes:
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()
            tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]
            author_link = quote.find("a")["href"] if quote.find("a") else None
            author_url = f"{base_url}{author_link}" if author_link else "N/A"
            scrape_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Fetch author details concurrently
            if author_url != "N/A":
                author_futures[author_url] = executor.submit(fetch_author_details, author_url)
            
            quote_data = {
                    'quote_id': str(uuid.uuid4()),  # New Unique Quote ID
                    'text': text,
                    'author': author,
                    'author_url': author_url,
                    'tags': tags,
                    'scrape_time': scrape_time,
                    'sentiment': get_sentiment(text),  # New Sentiment Analysis
                    'length': get_quote_length(text),  # New Feature: Quote Length
                    'word_count': get_word_count(text),  # New Feature: Word Count
                    'source': base_url,  # New Feature: Source URL
                    'language': detect_language(text),  # New Feature: Quote Language
                    'gender': get_author_gender(author),  # New Feature: Author's Gender
                    'nationality': get_author_nationality(author),  # New Feature: Author's Nationality
                    'occupation': get_author_occupation(author_url)  # New Feature: Author's Occupation
}

            quotes_list.append(quote_data)
    
    # Retrieve author details after threading
    for quote in quotes_list:
        if quote['author_url'] in author_futures:
            birth_date, birth_place = author_futures[quote['author_url']].result()
            quote['birth_date'] = birth_date
            quote['birth_place'] = birth_place
        else:
            quote['birth_date'], quote['birth_place'] = "N/A", "N/A"
    
    page_num += 1

# Insert data into SQLite
for quote in quotes_list:
    cursor.execute("""
        INSERT INTO quotes (quote_id, text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, source, language, gender, nationality, occupation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['quote_id'], quote['text'], quote['author'], quote['author_url'], quote['birth_date'], quote['birth_place'], ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['source'], quote['language'], quote['gender'], quote['nationality'], quote['occupation']))
    conn.commit()

conn.close()

# Saving data to files
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

# Save to CSV
with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["quote_id", "text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "source", "language", "gender", "nationality", "occupation"]
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
