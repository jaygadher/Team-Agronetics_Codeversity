import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import io
import base64
import json
from datetime import datetime

# Import configuration
from config import config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Ensure upload and detection results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_RESULTS'], exist_ok=True)

# Load the model
try:
    from model_loader import load_model
    model = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the model
model_path = 'plant_disease_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define plant disease classes - exact match from your notebook
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Define the exact model architecture from your notebook
class MyNN(nn.Module):
    def __init__(self, ip_features=3, num_classes=33):
        super().__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(ip_features, 16, kernel_size=3, padding=1),  # (16, 256, 256)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (16, 128, 128)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # (32, 128, 128)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # (128, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),         # (256, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # (256, 8, 8)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model with debug information
print(f"Loading model from: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")

try:
    # First try: load as state dict with correct architecture
    print("Attempting to load model state dict...")
    model = MyNN(ip_features=3, num_classes=len(disease_classes))
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    print(f"State dict keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model loaded successfully with correct architecture!")
    
except Exception as e:
    print(f"Failed to load as state dict: {e}")
    try:
        # Second try: load as complete model
        print("Attempting to load complete model...")
        model = torch.load(model_path, map_location=device)
        if hasattr(model, 'eval'):
            model.eval()
            model.to(device)
            print("Model loaded successfully as complete model")
        else:
            raise Exception("Loaded object doesn't have eval method")
    except Exception as e2:
        print(f"Failed to load complete model: {e2}")
        # Third try: create a fallback model
        print("Creating a fallback model...")
        model = MyNN(ip_features=3, num_classes=len(disease_classes))
        model.eval()
        model.to(device)

# Image preprocessing - match your training setup
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Match your training size
    transforms.ToTensor(),
    # No normalization - match your training setup
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def predict_disease(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        predicted_class = disease_classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        top3_predictions = []
        for i in range(3):
            top3_predictions.append({
                'class': disease_classes[top3_indices[i].item()],
                'confidence': top3_prob[i].item()
            })
        
        return predicted_class, confidence_score, top3_predictions
        
    except Exception as e:
        return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/disease-info')
def disease_info():
    return render_template('disease-info.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, confidence, top3 = predict_disease(filepath)
            
            if predicted_class:
                # Clean up uploaded file
                os.remove(filepath)
                
                # Create response data
                response_data = {
                    'success': True,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'top3': top3
                }
                
                return jsonify(response_data)
            else:
                return jsonify({'error': 'Failed to process image'})
                
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
    else:
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'})

def create_app():
    return app

def create_app():
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*50)
    print("Plant Disease Detection App")
    print("="*50)
    print(f"Environment: {'Development' if app.config['DEBUG'] else 'Production'}")
    print(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"Model path: {os.path.abspath(app.config['MODEL_PATH'])}")
    print("="*50 + "\n")
    
    if model is None:
        print("WARNING: Model failed to load. Some features may not work correctly.")
    
    print(f"Starting server on http://localhost:{port}")
    print("Press Ctrl+C to stop the server\n")
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
