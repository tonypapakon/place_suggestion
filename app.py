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
if not API_KEY:
    raise ValueError('No Google API key provided.')

#Categories for classification
CATEGORIES = ['service quality', 'value for money', 'food quality', 'ambiance']

#Load the classifier
classifier = pipeline(model="facebook/bart-large-mnli")

def geocode_address(address, api_key):
    """Geocode an address to latitude and longitude.

    Args:
        address (_type_): _description_
        api_key (_type_): _description_
    """
    
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    elif data['status'] == 'ZERO_RESULTS':
        raise ValueError('No results found for the specified address.')
    else:
        raise ValueError(f'Geocoding failed with status: {data["status"]}')

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        category = request.form['category']
        try:
            # Determine location
            if 'lat' in request.form and 'lng' in request.form and request.form['lat'] and request.form['lng']:
                lat = float(request.form['lat'])
                lng = float(request.form['lng'])
            else:
                location_text = request.form['location_text']
                if not location_text:
                    return render_template('results.html', error='No location provided.')
                lat, lng = geocode_address(location_text, API_KEY)
                
            # Perform a nearby search
            radius = 1500  # 1.5 km
            url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type={category}&key={API_KEY}'
            response = requests.get(url)
            data = response.json()
            if data['status'] != 'OK':
                return render_template('results.html', error='No places found.')
            places = data['results'][:3]  # Take first 3 places
            
            # Process each place
            place_data = []
            for place in places:
                place_id = place['place_id']
                name = place['name']
                # Extract latitude and longitude
                lat = place['geometry']['location']['lat']
                lng = place['geometry']['location']['lng']
                reviews = fetch_reviews(place_id, API_KEY)
                if not reviews:
                    continue
                overall_ratings = []
                category_counts = Counter()
                classify_reviews = []
                for review in reviews:
                    rating = review.get('rating', 'N/A')
                    if isinstance(rating, (int, float)):
                        overall_ratings.append(rating)
                        if rating >= 4:
                            category = classify_review(review['text'])
                            if category in CATEGORIES:
                                category_counts[category] += 1
                            else:
                                category = 'N/A'
                        else:
                            category = 'N/A'
                    else:
                        category = 'N/A'
                    
                    # Store all reviews for display            
                    classify_reviews.append({
                        'text': review['text'],
                        'rating': rating,
                        'category': category
                    })
                # Calculate average rating
                overall_avg = sum(overall_ratings) / len(overall_ratings) if overall_ratings else None

                # Best category
                most_common_category = category_counts.most_common(1)[0][0] if category_counts else 'N/A'
                # Add place data including lat and lng
                place_data.append({
                    'name': name,
                    'lat': lat,  # New: latitude
                    'lng': lng,  # New: longitude
                    'overall_avg': overall_avg,
                    'category': most_common_category,
                    'reviews': classify_reviews
                })
            return render_template('results.html', places=place_data, api_key=API_KEY)           
        except ValueError as e:
            return render_template('results.html', error=str(e))
        except Exception as e:
            return render_template('results.html', error=f"An error occurred: {str(e)}")
        
    # GET requests
    return render_template('index.html', api_key = API_KEY)

#Run app
if __name__ == '__main__':
    app.run(debug=True)