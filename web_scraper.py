import requests
from bs4 import BeautifulSoup

# Function to fetch content from a URL
def fetch_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        else:
            print(f"Failed to retrieve the webpage: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Function to find and follow the "Next" page link
def get_next_page(soup):
    next_page = soup.find('a', text='Next')
    if next_page:
        return next_page['href']
    return None

# Function to extract specific information from the webpage
def extract_data(soup):
    if soup:
        title = soup.title.string if soup.title else "No Title Found"
        meta_description = soup.find('meta', attrs={'name': 'description'})
        meta_description_content = meta_description['content'] if meta_description else "No Meta Description Found"
        
        paragraphs = soup.find_all('p')
        text_content = ""
        for para in paragraphs:
            text_content += para.get_text() + "\n"
        
        links = soup.find_all('a', href=True)
        link_content = "\nLinks Found:\n"
        for link in links:
            link_content += link['href'] + "\n"
        
        images = soup.find_all('img', src=True)
        image_content = "\nImage URLs Found:\n"
        for img in images:
            image_content += img['src'] + "\n"
        
        return title, meta_description_content, text_content, link_content, image_content
    else:
        print("No content to extract.")
        return None, None, None, None, None

# 🆕 New function to extract headings
def extract_headings(soup):
    headings = []
    for tag in ['h1', 'h2', 'h3']:
        found = soup.find_all(tag)
        for item in found:
            headings.append(f"{tag.upper()}: {item.get_text(strip=True)}")
    return "\n".join(headings)

# Function to save extracted content to a file
def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
        print(f"Content saved to {filename}")

# Main function to run the scraper
def main():
    url = input("Enter the URL to scrape: ")
    soup = fetch_url(url)
    
    if not soup:
        return
    
    all_content = ""
    while soup:
        title, meta_description, text_content, link_content, image_content = extract_data(soup)

        # 🆕 Add extracted headings
        headings = extract_headings(soup)
        all_content += "\nHeadings:\n" + headings + "\n"

        if title or meta_description or text_content or link_content or image_content:
            all_content += f"Title: {title}\nMeta Description: {meta_description}\n\n"
            all_content += text_content + "\n" + link_content + "\n" + image_content
        
        next_page_url = get_next_page(soup)
        if next_page_url:
            print(f"Following the next page: {next_page_url}")
            soup = fetch_url(next_page_url)
        else:
            break

    filename = input("Enter the filename to save the content (e.g., output.txt): ")
    save_to_file(all_content, filename)

if __name__ == "__main__":
    main()
