import requests
import pandas as pd
from googleapiclient.discovery import build
import matplotlib.pyplot as plt

# Your API key and CX (Search Engine ID)
API_KEY = 'AIzaSyDYdFdKTRCtSYgfAGIgaPtEfplLiYSuYK8'  
CX = 'd2806f352318d45d7'         


term_1 = [
    "Request for Proposal",
    "RFP",b     
    "Request for Bid", 
    "RFB",
    "Request for Quotation",       
    "RFQ",
    "Invitation to Tender",
    "ITT",
    "Call for Proposals"
]

term_2 = [
    "Bad debt recovery",
    "Debt collection",
    "Accounts receivable collection",
    "Revenue recovery",
    "Collection agency",
    "Debt recovery services",
    "Credit management",
    "Delinquent accounts",
    "Charge-off recovery",
    "Skip tracing",
    "Asset recovery",
    "Related financial terms",
    "Accounts receivable management",
    "AR optimization",
    "Cash flow improvement",
    "Credit risk mitigation",
    "Financial recovery",
    "Debt portfolio management",
    "First-party collections",
    "Third-party collections",
    "Early-stage collections",
    "Late-stage collections",
    "Legal collections",
    "Consumer collections",
    "Commercial collections",
    "Healthcare collections"
]

#  search using Google Custom Search API
def google_search(query, api_key, cse_id, num_results=10, start=1):
   
    service = build("customsearch", "v1", developerKey=api_key)
    
    # Call the API to get search results 
    
    result = service.cse().list(q=query, cx=cse_id, num=num_results, start=start, dateRestrict='Y1').execute()

    # Extracting results
    search_results = result.get('items', [])
    return [{'title': item.get('title'), 'url': item.get('link')} for item in search_results]

# Concatenating each term 
queries = [f"{t1} {t2}" for t1 in term_1 for t2 in term_2]

# Joining all queries into one string, separated by OR to make a single query
query_string = " OR ".join(queries)

# Retrieve and store all search results
results = []

# google custome search API only returns upto 10 results. 
#  so need to paginate to get 100 results (10 results per page) 
for i in range(1, 101, 10):  
    results_batch = google_search(query_string, API_KEY, CX, num_results=10, start=i)
    results.extend(results_batch)

# convert the results into a DataFrame for easy manipulation
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('google_search_results_filtered_2023_2024.csv', index=False)


print(df) 

# optional graph new
df['title'].value_counts().plot(kind='bar')
plt.show() 