import requests
import os
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter
from flask import Flask, request, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# Initialize Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key for session management

# Load Google's API key from .env
load_dotenv()
API_KEY = os.getenv('G_API_KEY')
if not API_KEY:
    raise ValueError('No Google API key provided in .env.')

# Categories for classification
CATEGORIES = ['service quality', 'value for money', 'food quality', 'ambiance']

# Load the classifier
classifier = pipeline(model="facebook/bart-large-mnli")

# --- Database Setup ---

def get_db_connection():
    """Connect to the SQLite database."""
    conn = sqlite3.connect('feedback.db')
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    return conn

def init_db():
    """Initialize the database with the user_feedback table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            user_id TEXT,
            place_id TEXT,
            liked BOOLEAN,
            category TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Run database initialization when the app starts
init_db()

# --- End Database Setup ---

def geocode_address(address, api_key):
    """Geocode an address to latitude and longitude."""
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
    """Fetch reviews for a given place ID from Google Maps API."""
    url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&reviews_no_translation=false&key={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'result' in data and 'reviews' in data['result']:
        reviews = data['result']['reviews']
        filtered_reviews = [review for review in reviews if review.get('text', '').strip()]
        return filtered_reviews[:max_reviews]
    return []

def classify_review(review_text):
    """Classify the review text into a category."""
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
            radius = 1000  # 1 km
            url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type={category}&key={API_KEY}'
            response = requests.get(url)
            data = response.json()
            if data['status'] != 'OK':
                return render_template('results.html', error='No places found.')
            
            # Get number of places from form (default to 5)
            max_places = int(request.form.get('max_places', 5))
            places = data['results'][:max_places]
            
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
                            cat = classify_review(review['text'])
                            if cat in CATEGORIES:
                                category_counts[cat] += 1
                            else:
                                cat = 'N/A'
                        else:
                            cat = 'N/A'
                    else:
                        cat = 'N/A'
                    classify_reviews.append({
                        'text': review['text'],
                        'rating': rating,
                        'category': cat
                    })
                overall_avg = sum(overall_ratings) / len(overall_ratings) if overall_ratings else None
                most_common_category = category_counts.most_common(1)[0][0] if category_counts else 'N/A'
                place_data.append({
                    'name': name,
                    'lat': lat,
                    'lng': lng,
                    'overall_avg': overall_avg,
                    'category': most_common_category,
                    'reviews': classify_reviews,
                    'place_id': place_id  # Added for feedback
                })
            
            # Personalize: re-rank places based on feedback
            user_id = session.get('user_id', 'default_user')  # Temporary user ID
            liked_categories = get_liked_categories(user_id)
            if liked_categories:
                place_data.sort(key=lambda p: compute_preference_score(p, liked_categories), reverse=True)
                
            # Build content model
            place_ids, embeddings, vectorizer = build_content_model(place_data)
            user_liked_texts = get_liked_texts(user_id, place_data)
            recommended_places = recommend_places_ml(place_ids, user_liked_texts, vectorizer, embeddings)
            
            return render_template('results.html', places=place_data, api_key=API_KEY, recommended_places=recommended_places)
        except ValueError as e:
            return render_template('results.html', error=str(e))
        except Exception as e:
            return render_template('results.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html', api_key=API_KEY)

def compute_preference_score(place, liked_categories):
    """Compute a preference score for a place based on user feedback."""
    score = 0
    if place['category'] in liked_categories:
        score += 1
    if place['overall_avg']:
        score += place['overall_avg'] / 5  # Normalize rating
    return score

def build_content_model(places):
    """Build a TF-IDF model for place reviews."""
    corpus = []
    place_ids = []
    for p in places:
        reviews_text = " ".join([r['text'] for r in p['reviews']])
        corpus.append(reviews_text)
        place_ids.append(p['name'])
    vectorizer = TfidfVectorizer(stop_words='english')
    embeddings = vectorizer.fit_transform(corpus)
    return place_ids, embeddings, vectorizer

def recommend_places_ml(current_places, user_liked_texts, vectorizer, embeddings):
    """Recommend places using ML based on user feedback."""
    liked_embedding = vectorizer.transform([" ".join(user_liked_texts)])
    similarity_scores = cosine_similarity(liked_embedding, embeddings)[0]
    recommendations = sorted(zip(current_places, similarity_scores), key=lambda x: x[1], reverse=True)
    return [rec[0] for rec in recommendations]

# --- Database Functions ---

def get_liked_categories(user_id):
    """Fetch liked categories from the database for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT category FROM user_feedback WHERE user_id = ? AND liked = 1', (user_id,))
    categories = [row['category'] for row in cursor.fetchall()]
    conn.close()
    return categories

def get_liked_texts(user_id, place_data):
    """Fetch texts of liked places from the database for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT place_id FROM user_feedback WHERE user_id = ? AND liked = 1', (user_id,))
    liked_place_ids = [row['place_id'] for row in cursor.fetchall()]
    conn.close()
    # Fetch review texts for liked places (simplified)
    liked_texts = []
    for place in place_data:  # Assuming place_data is accessible; adjust if needed
        if place['place_id'] in liked_place_ids:
            liked_texts.extend([r['text'] for r in place['reviews']])
    return liked_texts if liked_texts else ["positive feedback"]  # Fallback

# --- End Database Functions ---

@app.route('/rate', methods=['POST'])
def rate_place():
    """Handle user feedback submission."""
    place_id = request.form.get('place_id')
    category = request.form.get('category')
    user_id = session.get('user_id', 'default_user')  # Temporary user ID
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_feedback (user_id, place_id, liked, category) VALUES (?, ?, ?, ?)',
                   (user_id, place_id, True, category))
    conn.commit()
    conn.close()
    return render_template('results.html', places=[], api_key=API_KEY, error="Feedback recorded!")

# Run app
if __name__ == '__main__':
    app.run(debug=True)