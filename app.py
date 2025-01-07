import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model from a pickle file
with open('model (3).pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the form
        input_text1 = request.form.get('t1')
        input_text2 = request.form.get('t2')
        input_text3 = request.form.get('t3')

        # Convert input strings to float (or int, depending on your model's requirements)
        features = [
            float(input_text1),  # Convert to numeric
            float(input_text2),  # Convert to numeric
            float(input_text3)   # Convert to numeric
        ]

        # Prepare features for the model (if reshaping is required)
        final_features = np.array(features).reshape(1, -1)  # Reshape to match expected input format

        # Make a prediction
        prediction = model.predict(final_features)[0]  # Access the prediction result

        # Render the result.html with the prediction
        return render_template('result.html', prediction=f"The predicted output is: {prediction}")

    except ValueError:
        # Handle the case where inputs are not numeric
        return "Invalid input: Please enter numeric values for all inputs."

if __name__ == '__main__':
    app.run(debug=True)
