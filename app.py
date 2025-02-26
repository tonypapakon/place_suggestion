#Step 1. Fetch reviews from Google Maps using Places API
#Step 2. Process and categorize reviews
#Step 3. Handle multi-places
#Step 4. Save results on CSV
#Step 5. Make a web interface

import requests
import os
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv


#Google's API key for places.
load_dotenv()
API_KEY = os.getenv('G_API_KEY')


#List of place IDs to fetch reviews from Google Maps.
PLACE_IDS = ["ChIJlYH2uTACWBMRx0dkHjJUvfQ",
            "ChIJBxUOlzACWBMRZfIcqClCti0"
]

def fetch_reviews(place_id, api_key, max_reviews=5):
    """Fetches reviews for a given place ID from Google Maps API.

    Args:
        place_id (str): ID of the place. 
        api_key (str): Google API key.
        max_reviews (int, optional): Number of reviews that being checked. Defaults to 5 due to Google's API limitation.

    Returns:
        list: A list of reviews for the specified place.
    """
    
    # API request URL
    url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&reviews_no_translation=false&key={api_key}'
    # Send request to the API
    response = requests.get(url)
    data = response.json()

    if 'result' in data and 'reviews' in data['result']:
        reviews = data['result']['reviews']
        filtered_reviews = [review for review in reviews if review.get('text', '').strip()]
        return filtered_reviews[:max_reviews]
    return []

classifier = pipeline(model="facebook/bart-large-mnli")
CATEGORIES = ['service quality', 'value for money', 'food quality', 'ambiance']

def classify_review(review_text):
    """Classifies the review text into a category.

    Args:
        review_text (str): The text of the review to classify.

    Returns:
        str : The category of the review.
    """
    
    if review_text.strip():
        result = classifier(review_text, CATEGORIES)
        return result['labels'][0]
    return 'N/A'

def process_multi_place(place_ids,api_key):
    """Processes reviews for multiple places.

    Args:
        place_ids (list): List of place IDs to process.
        api_key (str): Google API key.
    """
    for place_id in place_ids:
        print(f"Processing reviews for place: {place_id}")
        reviews = fetch_reviews(place_id, api_key)
        if not reviews:
            print("No reviews found.")
            continue
    
        for review in tqdm(reviews, desc=f"Processing Reviews for {place_id}", unit="review"):
            category = classify_review(review['text'])
            print(f"Review: {review['text']}")
            print(f"Rating: {review.get('rating', 'N/A')}")
            print(f"Category: {category}")
            print("-"*400)

# Start processing reviews for all specified places            
process_multi_place(PLACE_IDS, API_KEY)