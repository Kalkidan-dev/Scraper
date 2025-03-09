import requests
from bs4 import BeautifulSoup
import csv
import json
import sqlite3
import time

# Database setup
conn = sqlite3.connect("quotes.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT UNIQUE,  -- Prevent duplicate entries
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

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

while True:
    URL = f"{base_url}/page/{page_num}/"
    
    try:
        response = requests.get(URL, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {URL}: {e}")
        break

    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")

    if not quotes:
        print("✅ Scraping complete. No more pages.")
        break  # Exit loop if no more quotes are found

    batch_data = []  # Store quotes for batch insert

    for quote in quotes:
        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()
        tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

        # Extract author page link
        author_link = quote.find("a")["href"] if quote.find("a") else None
        author_url = f"{base_url}{author_link}" if author_link else "N/A"

        # Extract author birth details
        birth_date, birth_place = "N/A", "N/A"
        if author_link:
            try:
                author_response = requests.get(author_url, headers=headers, timeout=10)
                author_response.raise_for_status()
                author_soup = BeautifulSoup(author_response.text, "html.parser")
                birth_date = author_soup.find("span", class_="author-born-date").get_text() if author_soup.find("span", class_="author-born-date") else "N/A"
                birth_place = author_soup.find("span", class_="author-born-location").get_text() if author_soup.find("span", class_="author-born-location") else "N/A"
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error fetching author details for {author}: {e}")

        quote_data = {
            'text': text,
            'author': author,
            'author_url': author_url,
            'birth_date': birth_date,
            'birth_place': birth_place,
            'tags': tags
        }
        quotes_list.append(quote_data)

        batch_data.append((text, author, author_url, birth_date, birth_place, ", ".join(tags)))

    # Insert into SQLite database (avoid duplicates)
    try:
        cursor.executemany("""
            INSERT OR IGNORE INTO quotes (text, author, author_url, birth_date, birth_place, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch_data)
        conn.commit()
    except Exception as db_error:
        print(f"⚠️ Database error: {db_error}")

    print(f"✅ Page {page_num} scraped. {len(quotes)} quotes found.")

    page_num += 1
    time.sleep(1)  # Avoid sending requests too quickly

# Close database connection
conn.close()

# Saving to a text file
with open("all_quotes.txt", "w", encoding="utf-8") as file:
    for quote in quotes_list:
        file.write(f"Quote: {quote['text']}\nAuthor: {quote['author']}\nAuthor URL: {quote['author_url']}\nBirth Date: {quote['birth_date']}\nBirth Place: {quote['birth_place']}\nTags: {', '.join(quote['tags'])}\n\n")

# Saving to a CSV file
with open("all_quotes.csv", "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Quote', 'Author', 'Author URL', 'Birth Date', 'Birth Place', 'Tags']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for quote in quotes_list:
        writer.writerow({'Quote': quote['text'], 'Author': quote['author'], 'Author URL': quote['author_url'], 'Birth Date': quote['birth_date'], 'Birth Place': quote['birth_place'], 'Tags': ', '.join(quote['tags'])})

# Saving to a JSON file
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

print("\n✅ All quotes have been saved to:")
print("  - all_quotes.txt")
print("  - all_quotes.csv")
print("  - all_quotes.json")
print("  - quotes.db (SQLite database)")
