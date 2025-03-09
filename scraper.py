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
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
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

        # Extract author birth details
        birth_date, birth_place = "N/A", "N/A"
        if author_link:
            try:
                author_response = requests.get(author_url, timeout=10)
                author_response.raise_for_status()
                author_soup = BeautifulSoup(author_response.text, "html.parser")
                birth_date = author_soup.find("span", class_="author-born-date").get_text() if author_soup.find("span", class_="author-born-date") else "N/A"
                birth_place = author_soup.find("span", class_="author-born-location").get_text() if author_soup.find("span", class_="author-born-location") else "N/A"
            except requests.exceptions.RequestException as e:
                print(f"Error fetching author details for {author}: {e}")

        quotes_list.append({
            'text': text,
            'author': author,
            'author_url': author_url,
            'birth_date': birth_date,
            'birth_place': birth_place,
            'tags': tags
        })

    page_num += 1

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

print("All quotes have been saved to all_quotes.txt, all_quotes.csv, and all_quotes.json.")
