from flask import Flask, render_template, request, jsonify
from dummy_prediction import get_dummy_prediction  # Dummy logic you created
from Upload_Image import Upload_Image_Prediction
from Capture_Image import Captured_Image_Prediction

app = Flask(__name__)

# Route to serve your main website (index.html)
@app.route('/')
def home():
    return render_template('index.html')  # Make sure this is in /templates

# Route for Upload Image
@app.route('/upload-handler', methods=['POST'])
def upload_handler():
    print("Upload handler triggered")
    result = Upload_Image_Prediction()
    return jsonify(result)

# Route for Capture Image
@app.route('/capture-handler', methods=['POST'])
def capture_handler():
    print("Capture handler triggered")
    result = Captured_Image_Prediction()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
