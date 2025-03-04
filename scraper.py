import requests
from bs4 import BeautifulSoup

URL = "http://quotes.toscrape.com"
response = requests.get(URL)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")

    for quote in quotes:
        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()
        tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

        print(f"Quote: {text}\nAuthor: {author}\nTags: {', '.join(tags)}\n")

else:
    print("Failed to retrieve the webpage.")
