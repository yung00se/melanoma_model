from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import image_normal
import cv2
#from model import predict_skin_lesion  # Import AI Model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the file first
    file.save(filepath)
    
    # Now read the saved file
    image = cv2.imread(filepath)
    
    # Check if image was loaded successfully
    if image is None:
        return jsonify({'error': 'Could not process the image'}), 400
        
    prediction = image_normal.predict_lesion(image=image)
    print("PREDICTION", prediction)
    
    prediction_result = None
    if prediction > 0.5:
        prediction_result = 'Malignant'
    else:
        prediction_result = 'Benign'
    
    return jsonify({'filename': filename, 'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
