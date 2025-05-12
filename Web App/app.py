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

app = Flask(__name__, static_url_path="",static_folder='assets',template_folder='templates')
CORS(app)

# app1 
model1 = load(r'population.joblib')


@app.route('/')
def index():
    return render_template('main_web_page.html')

@app.route('/population')
def population():
    return render_template('population.html')

@app.route('/predict_population', methods=['POST'])
def predict_():
    try:
        
        year = int(request.form['year'])
        
        
        input_data = np.array([[year]])
        
        
        prediction = model1.predict(input_data)
        
        
        return render_template('population.html', prediction=round(prediction[0]))
    except ValueError:
        return render_template('population.html', error="Invalid input. Please enter a valid year.")
    

# app2

model2 = load(r"decision_tree_model (4).joblib")


DB_FILE = 'diabetes_data.db'


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
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
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def map_age_to_bucket(age):
    if 18 <= age <= 24:
        return 1
    elif 25 <= age <= 29:
        return 2
    elif 30 <= age <= 34:
        return 3
    elif 35 <= age <= 39:
        return 4
    elif 40 <= age <= 44:
        return 5
    elif 45 <= age <= 49:
        return 6
    elif 50 <= age <= 54:
        return 7
    elif 55 <= age <= 59:
        return 8
    elif 60 <= age <= 64:
        return 9
    elif 65 <= age <= 69:
        return 10
    elif 70 <= age <= 74:
        return 11
    elif 75 <= age <= 79:
        return 12
    elif age >= 80:
        return 13
    else:
        raise ValueError("Age should be above 18")

@app.route('/lifestyle_form')
def form_page():
    return render_template('lifestyle_form.html')

@app.route('/lifestyle_result', methods=['POST'])
def result_():
    try:
       
        user_data = request.form.to_dict()

       
        encoding_map = {
            'Sex': {'M': 0, 'F': 1},
            'Smoker': {'No': 0, 'Yes': 1},
            'HighBP': {'No': 0, 'Yes': 1},
            'HighChol': {'No': 0, 'Yes': 1},
            'Stroke': {'No': 0, 'Yes': 1},
            'HeartDiseaseorAttack': {'No': 0, 'Yes': 1},
            'PhysActivity': {'No': 0, 'Yes': 1},
            'HvyAlcoholConsump': {'No': 0, 'Yes': 1}
        }
        

        
        user_data['Age'] = map_age_to_bucket(int(user_data['Age']))
        for key, mapping in encoding_map.items():
            user_data[key] = mapping.get(user_data[key], None)

        features = [
            user_data['Age'], 
            user_data['Sex'],  
            float(user_data['BMI']),
            user_data['Smoker'],  
            user_data['HighBP'],  
            user_data['HighChol'], 
            user_data['Stroke'],  
            user_data['HeartDiseaseorAttack'],  
            user_data['PhysActivity'], 
            user_data['HvyAlcoholConsump'],  
            float(user_data['GenHlth']),
            float(user_data['MentHlth']),
            float(user_data['PhysHlth'])
        ]

        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_data (
                Age, Sex, BMI, Smoker, HighBP, HighChol, Stroke,
                HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump,
                GenHlth, MentHlth, PhysHlth
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', features)
        conn.commit()
        conn.close()

        
        prediction = model2.predict([features])

        
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return render_template('lifestyle_result.html', result=result)
    except (ValueError, KeyError) as e:
        return f"Error in processing input data: {e}", 400
    except Exception as e:
        return f"Unexpected error: {e}", 500


# app3


model3 = load(r"random_forest_model.joblib")


encoding_map = {
    'GeneticMarkers': {'Positive': 1, 'Negative': 0},
    'Autoantibodies': {'Positive': 1, 'Negative': 0},
    'FamilyHistory': {'Yes': 1, 'No': 0},
    'EnvironmentalFactors': {'Present': 1, 'Absent': 0},
    'PhysicalActivity': {'low': 0, 'moderate': 1, 'high': 2},
    'DietaryHabits': {'Healthy': 1, 'Unhealthy': 0},
    'Ethnicity': {'high risk': 1, 'low risk': 0},
    'SocioeconomicFactors': {'low': 0, 'moderate': 1, 'high': 2},
    'SmokingStatus': {'smoker': 1, 'non-smoker': 0},
    'AlcoholConsumption': {'low': 0, 'moderate': 1, 'high': 2},
    'GlucoseToleranceTest': {'normal': 0, 'abnormal': 1},
    'HistoryOfPCOS': {'Yes': 1, 'No': 0},
    'PreviousGestationalDiabetes': {'Yes': 1, 'No': 0},
    'PregnancyHistory': {'normal': 0, 'complicated': 1},
    'CysticFibrosisDiagnosis': {'Yes': 1, 'No': 0},
    'SteroidUseHistory': {'Yes': 1, 'No': 0},
    'GeneticTesting': {'Positive': 1, 'Negative': 0},
    'LiverFunctionTests': {'normal': 0, 'abnormal': 1},
    'UrineTest': {
        'Protein Present': 3,
        'Normal': 2,
        'Glucose Present': 1,
        'Ketones Present': 0
    },
    'EarlyOnsetSymptoms': {'Yes': 1, 'No': 0},
}

@app.route('/medical_form')
def medical_form():
    return render_template('medical_form.html')

@app.route('/medical_result', methods=['POST'])
def result__():
    if request.method == 'POST':
        
        form_data = request.form.to_dict()

        
        encoded_data = []
        for key, value in form_data.items():
            if key in encoding_map:
                encoded_value = encoding_map[key][value]
                encoded_data.append(encoded_value)
            else:
                
                encoded_data.append(float(value))

        
        input_array = np.array([encoded_data])
        prediction = model3.predict(input_array)[0]

        
        return render_template('medical_result.html', prediction=prediction)
    

# app4
model4 = load(r"gestational.joblib")

@app.route('/gestation_form')
def gestation_form():
    return render_template('gestation_form.html')

@app.route('/gestational_predict', methods=['POST'])
def predict2():
    try:
        # Collect inputs from form
        age = float(request.form['age'])
        pregnancy_no = float(request.form['pregnancy_no'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = float(request.form['bmi'])
        heredity = float(request.form['heredity'])
        
        # Prepare the input for the model
        inputs = [age, pregnancy_no, weight, height, bmi, heredity]
        
        # Predict using the model
        prediction = model4.predict([inputs])[0]
        
        # Determine the result
        result = "has gestational diabetes" if prediction == 1 else "does not have gestational diabetes"
        return render_template('gestation_result.html', result=result)
    except Exception as e:
        return f"Error: {e}"

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

chatbot = Chatbot(r"chatbot_dataset.txt")

# Routes for Chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"response": "Please send a message."}), 400
    response = chatbot.get_response(user_message)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)