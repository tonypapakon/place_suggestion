# Place Suggestion

A web application that helps users find and analyze nearby cafes and bars using Google Maps API and natural language processing to categorize reviews.

## Features

- Search for nearby cafes and bars
- Use current location or enter a specific address
- View top 3 places with:
  - Location on map
  - Average rating
  - Categorized reviews
  - Most common category based on review analysis
- Review analysis categories:
  - Service quality
  - Value for money
  - Food quality
  - Ambiance

## Technologies Used

- Python 3.x
- Flask
- Google Maps Platform APIs
- Hugging Face Transformers
- BART Large MNLI model for review classification

## Prerequisites

Before running this application, you need:

1. Python 3.x installed
2. Google Cloud Platform account with billing enabled
3. Google Maps API key with the following APIs enabled:
   - Places API
   - Maps Embed API
   - Geocoding API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/place_suggestion.git
cd place_suggestion
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```bash
G_API_KEY=your_google_maps_api_key_here
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Either:
   - Enable location services and use current location
   - Enter a specific address
4. Select category (cafe or bar)
5. View results with maps and analyzed reviews

## Project Structure

```
place_suggestion/
├── app.py              # Main Flask application
├── templates/
│   ├── index.html     # Home page template
│   └── results.html   # Results page template
├── static/
│   └── style.css      # CSS styles
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # Project documentation
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Google Maps Platform for location services
- Hugging Face for the BART model
- Flask framework community