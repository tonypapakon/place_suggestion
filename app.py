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
        address (str): The address to geocode.
        api_key (str): Google API key.
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

def fetch_reviews(place_id, api_key, max_reviews=10):
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
        # Get user preferences from the form
        selected_preferences = request.form.getlist('preferences')  # List of selected categories
        
        try:
            # Determine location (unchanged)
            if 'lat' in request.form and 'lng' in request.form and request.form['lat'] and request.form['lng']:
                lat = float(request.form['lat'])
                lng = float(request.form['lng'])
            else:
                location_text = request.form['location_text']
                if not location_text:
                    return render_template('results.html', error='No location provided.')
                lat, lng = geocode_address(location_text, API_KEY)
                
            # Perform a nearby search (unchanged)
            radius = 1000  # 1 km
            url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type={category}&key={API_KEY}'
            response = requests.get(url)
            data = response.json()
            if data['status'] != 'OK':
                return render_template('results.html', error='No places found.')
            places = data['results'][:10]  # Take first 10 places
            
            # Compute user weights
            if selected_preferences:
                num_selected = len(selected_preferences)
                user_weights = {cat: 1.0 / num_selected if cat in selected_preferences else 0 for cat in CATEGORIES}
            else:
                user_weights = None  # No preferences selected
            
            # Process each place
            place_data = []
            for place in places:
                place_id = place['place_id']
                name = place['name']
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
                    classify_reviews.append({
                        'text': review['text'],
                        'rating': rating,
                        'category': category
                    })
                # Calculate average rating
                overall_avg = sum(overall_ratings) / len(overall_ratings) if overall_ratings else None
                
                # Best category (unchanged)
                most_common_category = category_counts.most_common(1)[0][0] if category_counts else 'N/A'
                
                # Compute place proportions
                total_high_rated = sum(category_counts.values())
                if total_high_rated > 0:
                    place_proportions = {cat: category_counts[cat] / total_high_rated for cat in CATEGORIES}
                else:
                    place_proportions = {cat: 0 for cat in CATEGORIES}  # No high-rated reviews
                
                # Compute personalized score
                if user_weights:
                    score = sum(user_weights[cat] * place_proportions[cat] for cat in CATEGORIES)
                else:
                    score = overall_avg if overall_avg is not None else 0  # Fallback to overall rating
                
                # Store place data with score
                place_data.append({
                    'name': name,
                    'lat': lat,
                    'lng': lng,
                    'overall_avg': overall_avg,
                    'category': most_common_category,
                    'reviews': classify_reviews,
                    'score': score  # New field for sorting
                })
            
            # Sort places by score (descending)
            place_data.sort(key=lambda x: x['score'], reverse=True)
            
            return render_template('results.html', places=place_data, api_key=API_KEY)           
        except ValueError as e:
            return render_template('results.html', error=str(e))
        except Exception as e:
            return render_template('results.html', error=f"An error occurred: {str(e)}")
        
    # GET requests (unchanged)
    return render_template('index.html', api_key=API_KEY)

#Run app
if __name__ == '__main__':
    app.run(debug=True)