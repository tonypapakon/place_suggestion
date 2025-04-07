<h1 align="center">
  <img src="static/logo_place_suggestion.png" alt="Logo" width="150" style="margin-bottom:20px;">
  <br>
  Place Suggestion
</h1>

Place Suggestion is a web application that helps users find and analyze nearby cafes and bars. Using the Google Maps API and natural language processing techniques, the app categorizes reviews and provides personalized recommendations based on user preferences.

## Features

- **Search for Nearby Cafes and Bars:** Use your current location or enter a specific address.
- **Visual Branding:** Updated with a custom logo displayed on the home screen and header bar.
- **Detailed Place Information:**
  - Location displayed on an interactive map.
  - Average rating from reviews.
  - Categorized review insights.
- **Review Analysis:** Reviews are analyzed into key categories:
  - Service quality
  - Value for money
  - Food quality
  - Ambiance
- **Personalized Recommendations:** The app re-ranks places based on your past feedback.
- **User Feedback:** Rate and provide feedback for places to improve future recommendations.

## Technologies Used

- Python 3.x
- Flask
- Google Maps Platform APIs
- Hugging Face Transformers (BART Large MNLI model for review classification)
- SQLite for lightweight storage
- TfidfVectorizer from scikit-learn for content modeling

## Prerequisites

Before running this application, you need:
1. Python 3.x installed
2. A Google Cloud Platform account with billing enabled
3. A Google Maps API key with the following APIs enabled:
   - Places API
   - Maps Embed API
   - Geocoding API

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tonypapakon/place_suggestion.git
   cd place_suggestion
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project root:**
   ```bash
   G_API_KEY=your_google_maps_api_key_here
   ```

5. **Add your logo:**
   - Save your logo image (with dimensions 447x447, e.g. `logo_place_suggestion.png`) in the `static` folder.
   - The logo appears in the header of both the home and results pages.

## Usage

1. **Start the Flask application:**
   ```bash
   python app.py
   ```

2. **Open a web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Using the App:**
   - Enable location services or manually enter an address.
   - Select your desired category (cafe or bar).
   - View detailed results with maps, reviews, and personalized recommendations.
   - Provide feedback to help improve suggestions.

## Project Structure

```
place_suggestion/
├── app.py              # Main Flask application
├── templates/
│   ├── index.html     # Home page template (includes logo & search form)
│   └── results.html   # Results page template (displays place info and logo)
├── static/
│   ├── style.css      # CSS styles
│   └── logo_place_suggestion.png  # Application logo for visual branding
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
└── README.md           # Project documentation
```

## Screenshots

### Light Mode

![Light Mode](screenshots/light_mode.png)

### Dark Mode

![Dark Mode](screenshots/dark_mode.png)

## Contributing

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Submit a pull request with your changes

## Acknowledgments

- Google Maps Platform for providing location services.
- Hugging Face for the BART model used in review classification.
- The Flask community for excellent documentation and support.
- All contributors and users who help improve Place Suggestion through feedback.