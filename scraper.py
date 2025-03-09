import requests
from bs4 import BeautifulSoup
import csv
import json

base_url = "http://quotes.toscrape.com"
page_num = 1
quotes_list = []

while True:
    URL = f"{base_url}/page/{page_num}/"
    
    try:
        response = requests.get(URL, timeout=10)  # Set timeout to avoid hanging requests
        response.raise_for_status()  # Raise HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {URL}: {e}")
        break

    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")

    if not quotes:
        break  # Exit loop if no more quotes are found

    for quote in quotes:
        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()
        tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

        # Extract author page link
        author_link = quote.find("a")["href"] if quote.find("a") else None
        author_url = f"{base_url}{author_link}" if author_link else "N/A"

        quotes_list.append({
            'text': text,
            'author': author,
            'author_url': author_url,
            'tags': tags
        })

    page_num += 1

# Saving to a text file
with open("all_quotes.txt", "w", encoding="utf-8") as file:
    for quote in quotes_list:
        file.write(f"Quote: {quote['text']}\nAuthor: {quote['author']}\nAuthor URL: {quote['author_url']}\nTags: {', '.join(quote['tags'])}\n\n")

# Saving to a CSV file
with open("all_quotes.csv", "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Quote', 'Author', 'Author URL', 'Tags']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for quote in quotes_list:
        writer.writerow({'Quote': quote['text'], 'Author': quote['author'], 'Author URL': quote['author_url'], 'Tags': ', '.join(quote['tags'])})

# Saving to a JSON file
with open("all_quotes.json", "w", encoding="utf-8") as jsonfile:
    json.dump(quotes_list, jsonfile, indent=4, ensure_ascii=False)

print("All quotes have been saved to all_quotes.txt, all_quotes.csv, and all_quotes.json.")
