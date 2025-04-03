import requests
from bs4 import BeautifulSoup
import csv
import json
import sqlite3
import time
from textblob import TextBlob
from collections import Counter

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
        popularity_score INTEGER,
        source TEXT
    )
""")
conn.commit()

# New feature: To track most common author and popular tags
author_counter = Counter()
tag_counter = Counter()

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
    """Verify the source of a quote using Wikipedia or Wikiquote."""
    search_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": text,
        "prop": "revisions",
        "rvprop": "content"
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check if the quote is found and has revisions
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "revisions" in page:
                return page["title"]  # Return the Wikipedia title of the quote's source
        return "Unknown"
    except Exception as e:
        log_error(f"Error fetching source for {text}: {e}")
        return "Unknown"

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
        
        # Update counters for author and tags
        author_counter[author] += 1
        tag_counter.update(tags)

        quote_data = {
            'text': text,
            'author': author,
            'author_url': author_url,
            'tags': tags,
            'scrape_time': scrape_time,
            'sentiment': get_sentiment(text),
            'length': get_quote_length(text),
            'word_count': get_word_count(text),
            'popularity_score': get_popularity_score(text),
            'source': get_verified_source(text)  # New feature: Source verification
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

# Save most common author and popular tags to a text file
with open("analysis.txt", "w") as analysis_file:
    analysis_file.write(f"Most Common Author: {author_counter.most_common(1)[0][0]}\n")
    analysis_file.write(f"Most Common Tags: {', '.join([tag for tag, _ in tag_counter.most_common(5)])}\n")
def get_language(text):
    """Detect the language of the quote."""
    try:
        return TextBlob(text).detect_language()
    except Exception as e:
        log_error(f"Error detecting language: {e}")
        return "Unknown"

# Modify database schema to add a language column
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN language TEXT;
""")
conn.commit()

# Define keyword-based categories
CATEGORIES = {
    "Motivation": ["dream", "goal", "success", "work", "hard", "achieve", "overcome"],
    "Love": ["love", "heart", "romance", "relationship", "affection"],
    "Life": ["life", "living", "experience", "journey", "existence"],
    "Success": ["success", "win", "achieve", "effort", "struggle"],
    "Wisdom": ["wisdom", "knowledge", "learning", "intelligence", "insight"]
}

def get_category(text):
    """Classify quote based on predefined categories."""
    text_lower = text.lower()
    for category, keywords in CATEGORIES.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "General"  # Default category if no match found

# Modify database schema to add a category column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN category TEXT;
""")
conn.commit()

# Add category classification in data collection
for quote in quotes_list:
    quote['category'] = get_category(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], quote['category']))
    conn.commit()

# Include category in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language", "category"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)
def detect_language(text):
    """Detect the language of the quote."""
    try:
        return TextBlob(text).detect_language()
    except Exception as e:
        log_error(f"Language detection failed for: {text[:30]}... Error: {e}")
        return "Unknown"

# Modify database schema to add a language column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN language TEXT;
""")
conn.commit()

# Add language detection in data collection
for quote in quotes_list:
    quote['language'] = detect_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()

# Include language in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

# Predefined themes with associated keywords
THEME_KEYWORDS = {
    "Motivation": ["inspire", "motivate", "dream", "achieve", "goal", "success"],
    "Love": ["love", "heart", "romance", "affection", "passion"],
    "Wisdom": ["wisdom", "knowledge", "learn", "intelligence", "truth"],
    "Life": ["life", "living", "experience", "journey", "existence"],
    "Success": ["success", "win", "achievement", "effort", "hard work"]
}

def classify_theme(text):
    """Classify the quote into a theme based on keywords."""
    text_lower = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return theme
    return "General"  # Default category if no keywords match

# Modify database schema to add a theme column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN theme TEXT;
""")
conn.commit()

# Add theme classification in data collection
for quote in quotes_list:
    quote['theme'] = classify_theme(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, theme)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], quote['theme']))
    conn.commit()

# Include theme in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language", "theme"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

# Add language detection in data collection
for quote in quotes_list:
    quote['language'] = get_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()

# Include language in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)
def get_author_bio(author_url):
    """Scrape author's biography from the author's page."""
    try:
        response = requests.get(author_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        bio = soup.find("span", class_="author-born-description")
        if bio:
            return bio.get_text(strip=True)
        return "Biography not available."
    except Exception as e:
        log_error(f"Error fetching bio for {author_url}: {e}")
        return "Error fetching biography."

# Modify database schema to add an author_bio column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN author_bio TEXT;
""")
conn.commit()

# Add author bio in data collection
for quote in quotes_list:
    author_bio = get_author_bio(quote['author_url'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, theme, author_bio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], quote['theme'], author_bio))
    conn.commit()

# Include author bio in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language", "theme", "author_bio"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

import tweepy

# Set up Twitter API client
def get_twitter_api():
    auth = tweepy.OAuthHandler("consumer_key", "consumer_secret")
    auth.set_access_token("access_token", "access_token_secret")
    api = tweepy.API(auth)
    return api

def get_social_media_mentions(text):
    """Fetch the number of mentions of the quote on Twitter."""
    api = get_twitter_api()
    try:
        # Search for the quote text on Twitter
        tweets = api.search(q=text, count=100, result_type="recent", lang="en")
        return len(tweets)
    except tweepy.TweepError as e:
        log_error(f"Error fetching social media mentions for {text}: {e}")
        return 0

# Modify database schema to add a social_media_mentions column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN social_media_mentions INTEGER;
""")
conn.commit()

# Add social media mentions in data collection
for quote in quotes_list:
    social_media_mentions = get_social_media_mentions(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, theme, author_bio, social_media_mentions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], quote['theme'], quote['author_bio'], social_media_mentions))
    conn.commit()

# Include social media mentions in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language", "theme", "author_bio", "social_media_mentions"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

# Simple rule-based genre classifier (can be improved with ML models)
def get_quote_genre(text):
    """Classify the genre of the quote based on its content."""
    motivational_keywords = ["inspire", "dream", "goal", "achieve", "success"]
    philosophical_keywords = ["life", "meaning", "existence", "truth", "reality"]
    love_keywords = ["love", "heart", "emotion", "affection", "passion"]

    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in motivational_keywords):
        return "Motivational"
    elif any(keyword in text_lower for keyword in philosophical_keywords):
        return "Philosophical"
    elif any(keyword in text_lower for keyword in love_keywords):
        return "Love"
    else:
        return "Unknown"

# Modify database schema to add a genre column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN genre TEXT;
""")
conn.commit()

# Add genre classification in data collection
for quote in quotes_list:
    genre = get_quote_genre(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, theme, author_bio, social_media_mentions, genre)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], quote['theme'], quote['author_bio'], quote['social_media_mentions'], genre))
    conn.commit()

# Include genre in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language", "theme", "author_bio", "social_media_mentions", "genre"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Ensures consistent results

def get_quote_language(text):
    """Detect the language of the quote."""
    try:
        return detect(text)
    except Exception as e:
        log_error(f"Error detecting language for {text}: {e}")
        return "Unknown"

# Modify database schema to add a language column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN language TEXT;
""")
conn.commit()

# Add language detection in data collection
for quote in quotes_list:
    language = get_quote_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], language))
    conn.commit()

# Include language in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)
def get_character_count(text):
    """Calculate the number of characters in the quote."""
    return len(text)

# Modify database schema to add a character_count column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN character_count INTEGER;
""")
conn.commit()

# Character count tracking
char_count_counter = Counter()

# Add character count in data collection
for quote in quotes_list:
    character_count = get_character_count(quote['text'])
    char_count_counter[character_count] += 1  # Track character count distribution

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, character_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote.get("language", "Unknown"), character_count))
    conn.commit()

# Include character count in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language", "character_count"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

# Save character count analysis
with open("character_count_analysis.txt", "w") as analysis_file:
    most_common_lengths = char_count_counter.most_common(5)
    analysis_file.write("Most Common Quote Character Lengths:\n")
    for length, count in most_common_lengths:
        analysis_file.write(f"{length} characters: {count} occurrences\n")

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Ensures consistent language detection results

def get_language(text):
    """Detect the language of the quote."""
    try:
        return detect(text)
    except:
        return "Unknown"

# Modify database schema to add a language column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN language TEXT;
""")
conn.commit()

# Language tracking
language_counter = Counter()

# Add language detection in data collection
for quote in quotes_list:
    language = get_language(quote['text'])
    language_counter[language] += 1  # Track language distribution

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], language))
    conn.commit()

# Include language in JSON and CSV
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

with open("all_quotes.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["text", "author", "author_url", "birth_date", "birth_place", "tags", "scrape_time", "sentiment", "length", "word_count", "popularity_score", "source", "language"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for quote in quotes_list:
        writer.writerow(quote)

# Save language analysis
with open("language_analysis.txt", "w") as analysis_file:
    most_common_languages = language_counter.most_common(5)
    analysis_file.write("Most Common Quote Languages:\n")
    for lang, count in most_common_languages:
        analysis_file.write(f"{lang}: {count} occurrences\n")
import requests

def verify_attribution(text, author):
    """Verify if a quote is correctly attributed to the author using Wikiquote."""
    search_url = "https://en.wikiquote.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": f"{text} {author}"
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # If a relevant result is found, assume attribution is verified
        search_results = data.get("query", {}).get("search", [])
        return bool(search_results)
    except Exception as e:
        log_error(f"Error verifying attribution for {text}: {e}")
        return False  # Assume false if verification fails

# Modify database schema to add an attribution_verified column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN attribution_verified BOOLEAN;
""")
conn.commit()

# Attribution tracking
misattributed_quotes = []

# Add verification in data collection
for quote in quotes_list:
    is_verified = verify_attribution(quote['text'], quote['author'])
    
    if not is_verified:
        misattributed_quotes.append((quote['text'], quote['author']))

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, attribution_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], is_verified))
    conn.commit()

# Save misattributed quotes analysis
with open("misattributed_quotes.txt", "w") as misattr_file:
    misattr_file.write("Misattributed Quotes Detected:\n")
    for text, author in misattributed_quotes:
        misattr_file.write(f'"{text}" - {author} (Possibly Misattributed)\n')
from textblob import TextBlob

def detect_language(text):
    """Detect the language of a quote."""
    try:
        return TextBlob(text).detect_language()
    except Exception as e:
        log_error(f"Error detecting language for quote: {e}")
        return "unknown"

# Modify database schema to add a language column (Run once)
cursor.execute("""
    ALTER TABLE quotes ADD COLUMN language TEXT;
""")
conn.commit()

# Language tracking
language_counter = Counter()

# Add language detection in data collection
for quote in quotes_list:
    quote_language = detect_language(quote['text'])
    language_counter[quote_language] += 1

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote_language))
    conn.commit()

# Save language distribution analysis
with open("language_analysis.txt", "w") as lang_file:
    lang_file.write("Most Common Quote Languages:\n")
    for lang, count in language_counter.most_common(5):
        lang_file.write(f"{lang}: {count} quotes\n")
def get_complexity_score(text):
    """Calculate the complexity score based on average word length."""
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

# Update data collection
for quote in quotes_list:
    quote['complexity_score'] = round(get_complexity_score(quote['text']), 2)

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, complexity_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['complexity_score']))
    conn.commit()
def is_first_letter_capital(text):
    """Check if the first letter of the quote is capitalized."""
    return "Yes" if text and text[0].isupper() else "No"

# Update data collection
for quote in quotes_list:
    quote['first_letter_capitalized'] = is_first_letter_capital(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, first_letter_capitalized)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['first_letter_capitalized']))
    conn.commit()
import textstat

def get_readability_score(text):
    """Calculate Flesch Reading Ease score for the quote."""
    return round(textstat.flesch_reading_ease(text), 2)

# Update data collection
for quote in quotes_list:
    quote['readability_score'] = get_readability_score(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, readability_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['readability_score']))
    conn.commit()
def detect_language(text):
    """Detect the language of the quote."""
    try:
        return detect(text)
    except:
        return "Unknown"

# Update data collection
for quote in quotes_list:
    quote['language'] = detect_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()
def get_complexity_score(text):
    """Calculate Automated Readability Index (ARI) for the quote."""
    word_count = get_word_count(text)
    char_count = sum(len(word) for word in text.split())
    
    if word_count == 0:  # Avoid division by zero
        return 0
    
    ari_score = 4.71 * (char_count / word_count) + 0.5 * (word_count / 1) - 21.43
    return round(ari_score, 2)

# Update data collection
for quote in quotes_list:
    quote['complexity_score'] = get_complexity_score(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, complexity_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['complexity_score']))
    conn.commit()

from rake_nltk import Rake

def extract_keywords(text):
    """Extract important keywords from the quote using RAKE (Rapid Automatic Keyword Extraction)."""
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return ", ".join(keywords[:5])  # Return top 5 keywords

# Update data collection
for quote in quotes_list:
    quote['keywords'] = extract_keywords(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['keywords']))
    conn.commit()
def is_inspirational(text):
    """Determine if a quote is inspirational based on sentiment and keywords."""
    positive_words = {"success", "dream", "hope", "believe", "achieve", "inspire", "motivate", "courage", "strength", "perseverance"}
    sentiment = get_sentiment(text)
    keywords = extract_keywords(text).split(", ")

    # A quote is considered inspirational if it's positive and contains key motivational words
    if sentiment == "Positive" and any(word in positive_words for word in keywords):
        return True
    return False

# Update data collection
for quote in quotes_list:
    quote['is_inspirational'] = is_inspirational(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, is_inspirational)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['is_inspirational']))
    conn.commit()
def is_duplicate(text, existing_quotes):
    """Check if a quote already exists in the database."""
    return text in existing_quotes

# Fetch existing quotes from the database
cursor.execute("SELECT text FROM quotes")
existing_quotes = {row[0] for row in cursor.fetchall()}

# Update data collection
for quote in quotes_list:
    quote['is_duplicate'] = is_duplicate(quote['text'], existing_quotes)

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, is_duplicate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['is_duplicate']))
    conn.commit()
def detect_language(text):
    """Detect the language of the quote."""
    try:
        return detect(text)
    except Exception:
        return "Unknown"

# Update data collection
for quote in quotes_list:
    quote['language'] = detect_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()
def classify_category(text):
    """Classify a quote into a predefined category based on keywords."""
    categories = {
        "Love": {"love", "heart", "romance", "relationship"},
        "Life": {"life", "journey", "experience", "living"},
        "Success": {"success", "achievement", "goal", "hard work"},
        "Wisdom": {"wisdom", "knowledge", "truth", "learning"},
        "Humor": {"funny", "humor", "laugh", "joke"}
    }
    
    words = set(text.lower().split())
    
    for category, keywords in categories.items():
        if words & keywords:  # If there's an intersection
            return category
    
    return "General"

# Update data collection
for quote in quotes_list:
    quote['category'] = classify_category(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['category']))
    conn.commit()
def is_inspirational(text):
    """Check if a quote is inspirational based on keywords and sentiment."""
    keywords = {"dream", "believe", "achieve", "success", "goal", "inspire", "motivate", "determination", "never give up", "overcome"}
    words = set(text.lower().split())

    sentiment = TextBlob(text).sentiment.polarity
    if words & keywords or sentiment > 0.2:  # If contains keyword or has positive sentiment
        return True
    return False

# Update data collection
for quote in quotes_list:
    quote['is_inspirational'] = is_inspirational(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, is_inspirational)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['is_inspirational']))
    conn.commit()

def detect_language(text):
    """Detect the language of a quote."""
    try:
        return detect(text)
    except Exception:
        return "Unknown"

# Update data collection
for quote in quotes_list:
    quote['language'] = detect_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()

def get_author_details(author_url):
    """Fetch author's birth date and birth place from their profile page."""
    try:
        response = requests.get(author_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        birth_date = soup.find("span", class_="author-born-date")
        birth_place = soup.find("span", class_="author-born-location")

        return birth_date.get_text() if birth_date else "Unknown", birth_place.get_text() if birth_place else "Unknown"
    except Exception as e:
        log_error(f"Error fetching author details: {e}")
        return "Unknown", "Unknown"
# Categorize quotes based on sentiment
categorized_quotes = {"Positive": [], "Negative": [], "Neutral": []}

for quote in quotes_list:
    categorized_quotes[quote["sentiment"]].append(quote)

# Save categorized quotes to separate JSON files
for sentiment, quotes in categorized_quotes.items():
    with open(f"{sentiment.lower()}_quotes.json", "w", encoding="utf-8") as jsonfile:
        json.dump(quotes, jsonfile, indent=4, ensure_ascii=False)

# Update data collection
for quote in quotes_list:
    birth_date, birth_place = get_author_details(quote['author_url'])
    quote['birth_date'] = birth_date
    quote['birth_place'] = birth_place

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], quote['birth_date'], quote['birth_place'], ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source']))
    conn.commit()


def is_duplicate(text):
    """Check if the quote already exists in the database."""
    cursor.execute("SELECT COUNT(*) FROM quotes WHERE text = ?", (text,))
    return cursor.fetchone()[0] > 0

# Update data collection
for quote in quotes_list:
    if is_duplicate(quote['text']):
        continue  # Skip duplicate quotes

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source']))
    conn.commit()

# Generate a report of the top 5 most quoted authors
top_authors = author_counter.most_common(5)

# Save report to a text file
with open("top_authors.txt", "w") as report_file:
    report_file.write("Top 5 Most Quoted Authors:\n")
    for author, count in top_authors:
        report_file.write(f"{author}: {count} quotes\n")

def detect_language(text):
    """Detect the language of a quote."""
    try:
        return detect(text)
    except Exception:
        return "Unknown"

# Update data collection
for quote in quotes_list:
    quote['language'] = detect_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], quote['birth_date'], quote['birth_place'], ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()

def get_author_biography(author_url):
    """Fetch the biography of an author from their profile page."""
    try:
        response = requests.get(author_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        bio_section = soup.find("div", class_="author-description")
        return bio_section.get_text(strip=True) if bio_section else "Biography not available"
    except Exception as e:
        log_error(f"Error fetching biography from {author_url}: {e}")
        return "Biography not available"

# Update data collection
for quote in quotes_list:
    quote['biography'] = get_author_biography(quote['author_url'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language, biography)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], quote['birth_date'], quote['birth_place'], ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language'], quote['biography']))
    conn.commit()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_quotes(quotes_list):
    """Find similar quotes using TF-IDF vectorization and cosine similarity."""
    texts = [quote['text'] for quote in quotes_list]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    similarity_matrix = cosine_similarity(vectorizer)

    similar_quotes = {}
    threshold = 0.7  # Adjust for stricter or looser matching
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i, j] > threshold:
                similar_quotes.setdefault(texts[i], []).append(texts[j])

    return similar_quotes

# Compute similar quotes
similar_quotes = find_similar_quotes(quotes_list)

# Save similar quotes report
with open("similar_quotes.txt", "w") as sim_file:
    sim_file.write("Similar Quotes Found:\n")
    for quote, sims in similar_quotes.items():
        sim_file.write(f"\nOriginal: {quote}\nSimilar:\n")
        for sim in sims:
            sim_file.write(f"- {sim}\n")

def detect_language(text):
    """Detect the language of a quote."""
    try:
        return detect(text)
    except Exception as e:
        log_error(f"Error detecting language for '{text}': {e}")
        return "Unknown"

# Update data collection
for quote in quotes_list:
    quote['language'] = detect_language(quote['text'])

    cursor.execute("""
        INSERT INTO quotes (text, author, author_url, birth_date, birth_place, tags, scrape_time, sentiment, length, word_count, popularity_score, source, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (quote['text'], quote['author'], quote['author_url'], "N/A", "N/A", ", ".join(quote['tags']), quote['scrape_time'], quote['sentiment'], quote['length'], quote['word_count'], quote['popularity_score'], quote['source'], quote['language']))
    conn.commit()

def analyze_quote_length_distribution(quotes_list):
    """Analyze the distribution of quote character lengths."""
    lengths = [quote['length'] for quote in quotes_list]
    
    # Generate histogram plot
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Quote Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Quote Lengths")
    plt.savefig("quote_length_distribution.png")

    # Save text-based summary
    with open("quote_length_analysis.txt", "w") as file:
        file.write(f"Total Quotes Analyzed: {len(lengths)}\n")
        file.write(f"Shortest Quote Length: {min(lengths)} characters\n")
        file.write(f"Longest Quote Length: {max(lengths)} characters\n")
        file.write(f"Average Quote Length: {sum(lengths) / len(lengths):.2f} characters\n")

# Run the analysis
analyze_quote_length_distribution(quotes_list)


# End time tracking and display execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Quotes saved to all_quotes.json, all_quotes.csv, quote_count.txt, and database.")
print(f"Analysis saved to analysis.txt.")
print(f"Total execution time: {execution_time:.2f} seconds")
