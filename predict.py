import numpy as np
import joblib

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_grade(study_hours, attendance, assignments, total_score):
    data = np.array([[study_hours, attendance, assignments, total_score]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    return prediction[0]