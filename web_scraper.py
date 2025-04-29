import requests
from bs4 import BeautifulSoup
import re
from collections import Counter


# Function to detect page language from the <html lang="..."> tag
def detect_page_language(soup):
    html_tag = soup.find('html')
    if html_tag and html_tag.has_attr('lang'):
        return html_tag['lang']
    return "Language not specified"

# Function to extract Twitter Card meta tags
def extract_twitter_meta_tags(soup):
    twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
    content = ""
    for tag in twitter_tags:
        name = tag.get('name', '')
        tag_content = tag.get('content', '')
        content += f"{name}: {tag_content}\n"
    return content.strip() if content else "No Twitter Card tags found"



# Function to count word frequency from visible text
def get_word_frequency(soup):
    text = soup.get_text().lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)  # Only words with 3+ letters
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)  # Top 10
    return "\n".join([f"{word}: {count}" for word, count in common_words])

# Function to extract data from all tables
def extract_tables(soup):
    tables_content = ""
    tables = soup.find_all('table')
    for idx, table in enumerate(tables, start=1):
        tables_content += f"\nTable {idx}:\n"
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['th', 'td'])
            row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
            tables_content += row_text + "\n"
    return tables_content.strip()


# Function to extract list items (ul and ol)
def extract_lists(soup):
    list_text = ""
    for list_tag in soup.find_all(['ul', 'ol']):
        items = list_tag.find_all('li')
        for item in items:
            list_text += f"- {item.get_text(strip=True)}\n"
        list_text += "\n"
    return list_text.strip()

# Function to extract all email addresses from the page
def extract_emails(soup):
    text = soup.get_text()
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return list(set(emails))  # remove duplicates

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

# Function to extract the canonical URL
def extract_canonical_url(soup):
    link_tag = soup.find('link', rel='canonical')
    if link_tag and link_tag.has_attr('href'):
        return link_tag['href']
    return "No canonical URL found"


# 🆕 New function to extract headings
def extract_headings(soup):
    headings = []
    for tag in ['h1', 'h2', 'h3']:
        found = soup.find_all(tag)
        for item in found:
            headings.append(f"{tag.upper()}: {item.get_text(strip=True)}")
    return "\n".join(headings)

# Function to extract all inline CSS styles
def extract_inline_styles(soup):
    styles = []
    elements_with_style = soup.find_all(style=True)
    for element in elements_with_style:
        styles.append(element['style'])
    return "\n".join(styles) if styles else "No inline styles found"

# Function to save extracted content to a file
def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
        print(f"Content saved to {filename}")
# Function to extract favicon URL
def extract_favicon(soup):
    icon_link = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
    if icon_link and icon_link.has_attr('href'):
        return icon_link['href']
    return "No favicon found"

# Function to extract JSON-LD structured data
def extract_json_ld(soup):
    scripts = soup.find_all('script', type='application/ld+json')
    json_ld_data = ""
    for script in scripts:
        json_ld_data += script.string.strip() + "\n\n" if script.string else ""
    return json_ld_data.strip() if json_ld_data else "No JSON-LD structured data found"

# Function to extract embedded YouTube video links
def extract_youtube_embeds(soup):
    youtube_links = []
    iframes = soup.find_all('iframe')
    for iframe in iframes:
        src = iframe.get('src', '')
        if 'youtube.com' in src or 'youtu.be' in src:
            youtube_links.append(src)
    return "\n".join(youtube_links) if youtube_links else "No YouTube embeds found"

# Function to extract PDF and document links
def extract_document_links(soup):
    doc_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if any(href.lower().endswith(ext) for ext in ['.pdf', '.docx', '.pptx', '.xlsx']):
            doc_links.append(href)
    return "\n".join(doc_links) if doc_links else "No document links found"


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

        emails = extract_emails(soup)
        if emails:
            all_content += "\nEmail Addresses Found:\n" + "\n".join(emails) + "\n"

        lists = extract_lists(soup)
        if lists:
            all_content += "\nLists Found:\n" + lists + "\n"

        tables = extract_tables(soup)
        if tables:
            all_content += "\nTables Found:\n" + tables + "\n"
            language = detect_page_language(soup)
        all_content += f"\nPage Language: {language}\n"

        word_freq = get_word_frequency(soup)
        all_content += "\nTop 10 Word Frequencies:\n" + word_freq + "\n"

        document_links = extract_document_links(soup)
        all_content += "\nDocument Links:\n" + document_links + "\n"

        inline_styles = extract_inline_styles(soup)
        all_content += "\nInline CSS Styles:\n" + inline_styles + "\n"

        

        json_ld = extract_json_ld(soup)
        all_content += "\nStructured Data (JSON-LD):\n" + json_ld + "\n"

        youtube_embeds = extract_youtube_embeds(soup)
        all_content += "\nEmbedded YouTube Videos:\n" + youtube_embeds + "\n"

        twitter_meta = extract_twitter_meta_tags(soup)
        all_content += "\nTwitter Card Tags:\n" + twitter_meta + "\n"

        canonical_url = extract_canonical_url(soup)
        all_content += f"\nCanonical URL: {canonical_url}\n"

        favicon_url = extract_favicon(soup)
        all_content += f"\nFavicon URL: {favicon_url}\n"

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
