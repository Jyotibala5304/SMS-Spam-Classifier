from flask import Flask, render_template, request
import joblib

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')



# Load the pre-trained model using joblib
model = joblib.load('model.pkl')

@app.route('/')
def index():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_text = request.form['text']
    
    # Make a prediction using the model
    prediction = model.predict([input_text])
    
    # Convert prediction to readable label
    label = 'Spam' if prediction[0] == 1 else 'Not Spam'
    
    # Render the result on the same page
    return render_template('index.html', prediction=label, text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
