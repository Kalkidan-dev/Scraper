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
        # Extract all paragraph text from the webpage
        paragraphs = soup.find_all('p')
        text_content = ""
        for para in paragraphs:
            text_content += para.get_text() + "\n"
        return text_content
    else:
        print("No content to extract.")
        return None

# Function to save extracted content to a file
def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
        print(f"Content saved to {filename}")

# Main function to run the scraper
def main():
    url = input("Enter the URL to scrape: ")
    soup = fetch_url(url)
    content = extract_data(soup)
    
    if content:
        # Save the content to a file
        filename = input("Enter the filename to save the content (e.g., output.txt): ")
        save_to_file(content, filename)

if __name__ == "__main__":
    main()
