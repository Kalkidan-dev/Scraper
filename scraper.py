import requests
from bs4 import BeautifulSoup
import csv
import json
import sqlite3
import concurrent.futures

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
        tags TEXT
    )
""")
conn.commit()

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
    except requests.exceptions.RequestException:
        return "N/A", "N/A"

while True:
    URL = f"{base_url}/page/{page_num}/"
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {URL}: {e}")
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
            
            # Fetch author details concurrently
            if author_url != "N/A":
                author_futures[author_url] = executor.submit(fetch_author_details, author_url)
            
            quote_data = {
                'text': text,
                'author': author,
                'author_url': author_url,
                'tags': tags
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
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], quote['birth_date'], quote['birth_place'], ", ".join(quote['tags'])))
    conn.commit()

conn.close()

# Saving data to files
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

print("Quotes saved to all_quotes.json and database.")
