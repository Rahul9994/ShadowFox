"""
Flask Backend API for Sentiment Analysis
Run this with: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the trained sentiment analysis model"""
    global model, tokenizer, device
    
    print("Loading model and tokenizer...")
    model_path = "./my_sentiment_model"
    
    # Check if trained model exists, otherwise use pretrained
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Loading trained model from {model_path}")
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    else:
        print("Trained model not found. Using pretrained model...")
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully on {device}")

@app.route('/')
def index():
    """Serve the HTML file"""
    return app.send_static_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get prediction
        predicted_class_id = logits.argmax().item()
        probabilities = torch.softmax(logits, dim=1)[0]
        
        label = "POSITIVE" if predicted_class_id == 1 else "NEGATIVE"
        confidence = probabilities[predicted_class_id].item()
        negative_prob = probabilities[0].item()
        positive_prob = probabilities[1].item()
        
        return jsonify({
            'sentiment': label,
            'confidence': round(confidence * 100, 2),
            'negative_probability': round(negative_prob * 100, 2),
            'positive_probability': round(positive_prob * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    print("\n" + "="*50)
    print("Sentiment Analysis API Server")
    print("="*50)
    print("Server starting on http://localhost:5000")
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
