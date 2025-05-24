from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # 4 classes
model.load_state_dict(torch.load('./model/vehicle_classification_model.pth', map_location=device))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names (ensure order matches training)
classes = ['bicycle', 'car', 'motorbike', 'truck']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load and transform image
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_index = torch.argmax(probabilities).item()
            predicted_class = classes[predicted_index]
            confidence = probabilities[predicted_index].item()

        # Determine confidence color
        if confidence >= 0.80:
            color = 'green'
        elif confidence >= 0.50:
            color = 'yellow'
        else:
            color = 'red'

        # Return response with prediction and confidence
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'color': color
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
