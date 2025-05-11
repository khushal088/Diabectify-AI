import random
import re
import os
import numpy as np
import sqlite3
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Shared Resources
DB_FILE = 'diabetes_data.db'
FAQ_FILE_PATH = r"C:\Users\khushal\Desktop\completenew\chatbot_dataset.txt"

# Initialize Models
model1 = load(r'joblib models\population.joblib')
model2 = load(r"joblib models\decision_tree_model (4).joblib")
model3 = load(r"joblib models\random_forest_model.joblib")
model4 = load(r"joblib models\gestational.joblib")


# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_data (
        PatientID INTEGER PRIMARY KEY AUTOINCREMENT,
        Age INT CHECK (Age > 0 AND Age <= 13),
        Sex TINYINT(1),
        BMI FLOAT CHECK (BMI > 0 AND BMI < 60),
        Smoker TINYINT(1),
        HighBP TINYINT(1),
        HighChol TINYINT(1),
        Stroke TINYINT(1),
        HeartDiseaseorAttack TINYINT(1),
        PhysActivity TINYINT(1),
        HvyAlcoholConsump TINYINT(1),
        GenHlth INT CHECK (GenHlth > 0 AND GenHlth <= 5),
        MentHlth INT CHECK (MentHlth >= 0 AND MentHlth <= 30),
        PhysHlth INT CHECK (PhysHlth >= 0 AND PhysHlth <= 30)
    )''')
    conn.commit()
    conn.close()

init_db()

# Chatbot Class
class Chatbot:
    def __init__(self, faq_file_path):
        self.greetings = ["hello", "hi", "hey"]
        self.greeting_responses = ["Hello! How can I assist you?", "Hi there!", "Greetings!"]
        self.default_responses = ["I'm here to help.", "What can I assist you with today?"]
        self.faq_responses = {}
        self.vectorizer = None
        self.faq_questions = []
        self.load_faqs(faq_file_path)

    def preprocess_text(self, text):
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    def load_faqs(self, faq_file_path):
        if not os.path.exists(faq_file_path):
            return
        with open(faq_file_path, "r", encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    question, answer = parts
                    processed_question = self.preprocess_text(question)
                    self.faq_responses[processed_question] = answer.strip()
        self.faq_questions = list(self.faq_responses.keys())
        self.vectorizer = TfidfVectorizer().fit(self.faq_questions)

    def find_best_match(self, user_input):
        if not self.vectorizer:
            return None
        user_input_vector = self.vectorizer.transform([self.preprocess_text(user_input)])
        faq_vectors = self.vectorizer.transform(self.faq_questions)
        similarities = cosine_similarity(user_input_vector, faq_vectors)
        max_sim_index = similarities.argmax()
        return self.faq_questions[max_sim_index] if similarities[0, max_sim_index] > 0.2 else None

    def get_response(self, user_input):
        if any(greet in self.preprocess_text(user_input) for greet in self.greetings):
            return random.choice(self.greeting_responses)
        best_match = self.find_best_match(user_input)
        return self.faq_responses.get(best_match, random.choice(self.default_responses))

chatbot = Chatbot(FAQ_FILE_PATH)

# Routes for Chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"response": "Please send a message."}), 400
    response = chatbot.get_response(user_message)
    return jsonify({"response": response})

# Routes for Machine Learning Models
@app.route('/population', methods=['GET', 'POST'])
def population():
    if request.method == 'POST':
        year = int(request.form['year'])
        prediction = model1.predict(np.array([[year]]))
        return render_template(r'C:\Users\khushal\Desktop\INTEL-INDICON\templates\population.html', prediction=round(prediction[0]))
    return render_template('population.html')

@app.route('/lifestyle_form', methods=['GET', 'POST'])
def lifestyle_form():
    if request.method == 'POST':
        # Handle form data processing and prediction
        pass  # Add the prediction logic here
    return render_template('lifestyle_form.html')

@app.route('/medical_form', methods=['GET', 'POST'])
def medical_form():
    if request.method == 'POST':
        # Handle medical form data processing and prediction
        pass  # Add the prediction logic here
    return render_template('medical_form.html')

@app.route('/gestation_form', methods=['GET', 'POST'])
def gestation_form():
    if request.method == 'POST':
        # Handle gestational diabetes form data processing and prediction
        pass  # Add the prediction logic here
    return render_template('gestation_form.html')

# Main Index Route
@app.route('/')
def index():
    return render_template('main_web_page.html')

if __name__ == '__main__':
    app.run(debug=True)
