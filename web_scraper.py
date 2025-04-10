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

# Function to extract specific information from the webpage (title, meta description, paragraphs, links, and images)
def extract_data(soup):
    if soup:
        # Extract the page title
        title = soup.title.string if soup.title else "No Title Found"
        
        # Extract the meta description
        meta_description = soup.find('meta', attrs={'name': 'description'})
        meta_description_content = meta_description['content'] if meta_description else "No Meta Description Found"
        
        # Extract all paragraph text from the webpage
        paragraphs = soup.find_all('p')
        text_content = ""
        for para in paragraphs:
            text_content += para.get_text() + "\n"
        
        # Extract all links from the webpage
        links = soup.find_all('a', href=True)
        link_content = "\nLinks Found:\n"
        for link in links:
            link_content += link['href'] + "\n"
        
        # Extract all image sources from the webpage
        images = soup.find_all('img', src=True)
        image_content = "\nImage URLs Found:\n"
        for img in images:
            image_content += img['src'] + "\n"
        
        return title, meta_description_content, text_content, link_content, image_content
    else:
        print("No content to extract.")
        return None, None, None, None, None

# Function to save extracted content to a file
def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
        print(f"Content saved to {filename}")

# Main function to run the scraper
def main():
    url = input("Enter the URL to scrape: ")
    soup = fetch_url(url)
    title, meta_description, text_content, link_content, image_content = extract_data(soup)
    
    if title or meta_description or text_content or link_content or image_content:
        # Save the title, meta description, text content, links, and images to a file
        filename = input("Enter the filename to save the content (e.g., output.txt): ")
        full_content = f"Title: {title}\nMeta Description: {meta_description}\n\n"
        full_content += text_content + "\n" + link_content + "\n" + image_content
        save_to_file(full_content, filename)

if __name__ == "__main__":
    main()
