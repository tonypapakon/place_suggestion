#Step 1. Fetch reviews from Google Maps using Places API
#Step 2. Process and categorize reviews
#Step 3. Handle multi-places
#Step 4. Save results on CSV
#Step 5. Make a web interface

#import pandas as pd
import requests
import os
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv


#Google's API key for places
load_dotenv()
API_KEY = os.getenv('G_API_KEY')


#Get the place's id from google maps
PLACE_ID = "ChIJlYH2uTACWBMRx0dkHjJUvfQ"

def fetch_reviews(place_id, api_key, max_reviews=5):
    url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&reviews_no_translation=false&key={api_key}'
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
    if review_text.strip():
        result = classifier(review_text, CATEGORIES)
        return result['labels'][0]
    return 'N/A'

#test the function
reviews = fetch_reviews(PLACE_ID, API_KEY)

for review in tqdm(reviews, desc="Processing Reviews", unit="review"):
    category = classify_review(review['text'])
    print(f"Review: {review['text']}")
    print(f"Rating: {review['rating']}")
    print(f"Category: {category}")
    print("---")