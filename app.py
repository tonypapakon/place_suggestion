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
from collections import Counter
from flask import Flask, request, render_template

#Initialize Flask
app = Flask(__name__)

#Google's API key for places.
load_dotenv()
API_KEY = os.getenv('G_API_KEY')

#Categories for classification
CATEGORIES = ['service quality', 'value for money', 'food quality', 'ambiance']

#Load the classifier
classifier = pipeline(model="facebook/bart-large-mnli")


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

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        place_id = request.form['place_id']
        try:
            #Fetch review
            reviews = fetch_reviews(place_id, API_KEY)
            if not reviews:
                return render_template('results.html', error = "No reviews found or invalid ID.")
            
            #Process reviews
            overall_ratings = []
            category_counts = Counter()
            classify_reviews = []
            
            for review in reviews:
                category = classify_review(review['text'])
                rating = review.get('rating', 'N/A')
                if isinstance(rating, (int, float)):
                    overall_ratings.append(rating)
                if category in CATEGORIES:
                    category_counts[category] += 1
                classify_reviews.append({
                    'text': review['text'],
                    'rating': rating,
                    'category': category
                })
                    
            #Cal avg rating
            overall_avg = sum(overall_ratings) / len(overall_ratings) if overall_ratings else None
            
            #Best category
            most_common_categoy = category_counts.most_common(1)[0][0] if category_counts else 'N/A'
            
            #Render results page with data
            return render_template('results.html',
                                   overall_avg=overall_avg,
                                   most_common_categoy=most_common_categoy,
                                   reviews=reviews)
            
        except Exception as e:
            return render_template('results.html', error = f"An error occurred: {str(e)}")
        
    #GET requests
    return render_template('index.html')

#Run app
if __name__ == '__main__':
    app.run(debug=True)