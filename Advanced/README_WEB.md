# Sentiment Analysis Web Application

A beautiful, modern web interface for sentiment analysis powered by DistilBERT.

## Features

- 🎨 Beautiful, modern UI with gradient backgrounds and animations
- ⚡ Real-time sentiment analysis
- 📊 Visual confidence indicators and probability scores
- 📱 Fully responsive design
- 🚀 Fast and efficient API backend

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

Run the training script to create the model:

```bash
python sentiment_analysis.py
```

This will create a `my_sentiment_model` directory with the trained model.

### 3. Start the Web Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Open in Browser

Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
Advanced/
├── app.py                 # Flask backend API
├── sentiment_analysis.py  # Training script
├── static/
│   ├── index.html        # Frontend HTML
│   ├── styles.css        # Beautiful CSS styles
│   └── script.js         # Frontend JavaScript
├── my_sentiment_model/   # Trained model (created after training)
└── requirements.txt      # Python dependencies
```

## Usage

1. Enter text in the text area
2. Click "Analyze Sentiment" or press Ctrl+Enter
3. View the sentiment (POSITIVE/NEGATIVE), confidence score, and probability breakdown

## API Endpoints

- `GET /` - Serves the HTML frontend
- `POST /api/predict` - Analyzes sentiment of provided text
- `GET /api/health` - Health check endpoint

## Customization

You can customize the appearance by editing `static/styles.css`. The color scheme uses CSS variables defined in the `:root` selector.

## Troubleshooting

- **Model not found**: Make sure you've run `sentiment_analysis.py` first to train and save the model
- **High CPU usage**: The model uses CPU threads limited to 2. Adjust in `app.py` if needed
- **Port already in use**: Change the port in `app.py` (line with `app.run()`)
