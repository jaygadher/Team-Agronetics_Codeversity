<<<<<<< HEAD
# Plant Disease Detection Web Application

A simple and user-friendly web application for detecting plant diseases using machine learning. Upload a photo of a plant leaf and get instant predictions with confidence scores.

## Features

- 🌱 **Easy Upload**: Drag & drop or click to upload plant leaf images
- 🔍 **AI Detection**: Uses your trained PyTorch model for accurate disease detection
- 📊 **Detailed Results**: Shows main prediction and top 3 possibilities with confidence scores
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- ⚡ **Fast Processing**: Quick analysis and results display

## Setup Instructions

### 1. Install Python Dependencies

First, make sure you have Python installed (3.7 or higher). Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

### 3. Use the Application

1. Open your web browser and go to `http://localhost:5000`
2. Upload an image of a plant leaf by:
   - Dragging and dropping the image onto the upload area, or
   - Clicking "Choose Image" to browse and select a file
3. Wait for the analysis to complete
4. View the disease prediction results with confidence scores

## Supported File Types

- PNG
- JPG
- JPEG
- GIF

## File Size Limit

Maximum file size: 16MB

## Model Information

The application uses your `plant_disease_model.pth` file to make predictions. It can detect 38 different plant diseases across various plant types including:

- Apple diseases
- Corn diseases
- Grape diseases
- Potato diseases
- Tomato diseases
- And many more...

## Troubleshooting

### Common Issues:

1. **"No module named 'torch'"**: Make sure you've installed all requirements with `pip install -r requirements.txt`

2. **Model loading errors**: Ensure your `plant_disease_model.pth` file is in the same directory as `app.py`

3. **Upload errors**: Check that your image file is under 16MB and in a supported format

4. **Prediction errors**: Make sure the uploaded image shows a clear view of plant leaves

## Customization

You can customize the application by:

- Modifying the disease classes in `app.py` if your model uses different categories
- Adjusting the model architecture in the `PlantDiseaseCNN` class
- Changing the styling in the HTML template
- Modifying the confidence thresholds

## Technical Details

- **Backend**: Flask (Python web framework)
- **ML Framework**: PyTorch
- **Image Processing**: PIL/Pillow
- **Frontend**: HTML, CSS, JavaScript
- **Model Format**: PyTorch (.pth)

## Support

If you encounter any issues, check the console output for error messages and ensure all dependencies are properly installed.
=======
# Team-Agronetics_Codeversity
>>>>>>> 86c41b8e6d6c62196b2c4602425febfa6f50c6dd
