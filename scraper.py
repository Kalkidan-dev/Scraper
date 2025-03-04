import requests
from bs4 import BeautifulSoup
import csv

URL = "http://quotes.toscrape.com"
response = requests.get(URL)

tag_filter = "love"  # Filter quotes by this tag

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")

    filtered_quotes = []

    for quote in quotes:
        tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]
        if tag_filter in tags:
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()

            filtered_quotes.append({
                'text': text,
                'author': author,
                'tags': tags
            })

    # Saving to a text file
    with open("filtered_quotes.txt", "w") as file:
        for quote in filtered_quotes:
            file.write(f"Quote: {quote['text']}\nAuthor: {quote['author']}\nTags: {', '.join(quote['tags'])}\n\n")

    # Saving to a CSV file
    with open("filtered_quotes.csv", "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Quote', 'Author', 'Tags']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for quote in filtered_quotes:
            writer.writerow({'Quote': quote['text'], 'Author': quote['author'], 'Tags': ', '.join(quote['tags'])})

    print(f"Filtered quotes with the tag '{tag_filter}' have been saved to filtered_quotes.txt and filtered_quotes.csv.")
else:
    print("Failed to retrieve the webpage.")
