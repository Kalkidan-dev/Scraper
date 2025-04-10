import requests
from bs4 import BeautifulSoup

# Function to fetch content from a URL
def fetch_url(url):
    try:
        # Send an HTTP request to the URL
        response = requests.get(url)
        
        # If request was successful, parse the HTML content
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        else:
            print(f"Failed to retrieve the webpage: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Function to extract specific information from the webpage
def extract_data(soup):
    if soup:
        # For example, extract all paragraph text from the webpage
        paragraphs = soup.find_all('p')
        for para in paragraphs:
            print(para.get_text())
    else:
        print("No content to extract.")

# Main function to run the scraper
def main():
    url = input("Enter the URL to scrape: ")
    soup = fetch_url(url)
    extract_data(soup)

if __name__ == "__main__":
    main()
