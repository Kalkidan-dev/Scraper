import requests
from bs4 import BeautifulSoup
import csv

URL = "http://quotes.toscrape.com"
response = requests.get(URL)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")

    # Saving to a text file
    with open("quotes.txt", "w") as file:
        for quote in quotes:
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()
            tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

            file.write(f"Quote: {text}\nAuthor: {author}\nTags: {', '.join(tags)}\n\n")

    # Saving to a CSV file
    with open("quotes.csv", "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Quote', 'Author', 'Tags']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for quote in quotes:
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()
            tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

            writer.writerow({'Quote': text, 'Author': author, 'Tags': ', '.join(tags)})

    print("Quotes have been saved to quotes.txt and quotes.csv.")
else:
    print("Failed to retrieve the webpage.")
