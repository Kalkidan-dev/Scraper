import requests
from bs4 import BeautifulSoup
import csv

base_url = "http://quotes.toscrape.com"
page_num = 1
quotes_list = []

while True:
    URL = f"{base_url}/page/{page_num}/"
    response = requests.get(URL)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        quotes = soup.find_all("div", class_="quote")

        for quote in quotes:
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()
            tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

            quotes_list.append({
                'text': text,
                'author': author,
                'tags': tags
            })

        next_page = soup.find("li", class_="next")
        if next_page:
            page_num += 1
        else:
            break
    else:
        print("Failed to retrieve the webpage.")
        break

# Saving to a text file
with open("all_quotes.txt", "w") as file:
    for quote in quotes_list:
        file.write(f"Quote: {quote['text']}\nAuthor: {quote['author']}\nTags: {', '.join(quote['tags'])}\n\n")

# Saving to a CSV file
with open("all_quotes.csv", "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Quote', 'Author', 'Tags']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for quote in quotes_list:
        writer.writerow({'Quote': quote['text'], 'Author': quote['author'], 'Tags': ', '.join(quote['tags'])})

print("All quotes have been saved to all_quotes.txt and all_quotes.csv.")
