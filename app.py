import streamlit as st
import pandas as pd
import joblib
import numpy as np
import zipfile

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("dataset.zip")

# Page Settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "Manual Prediction",
        "CSV Upload",
        "Model Performance",
        "About"
    ]
)

# ---------------- HOME ----------------

if page == "Home":

    st.title("🎓 Student Performance Prediction Dashboard")

    st.write("Welcome to the Machine Learning Deployment Dashboard.")

    st.write("### Features")

    st.write("- Student Grade Prediction")
    st.write("- Manual Prediction")
    st.write("- CSV Batch Prediction")
    st.write("- Model Performance")
    st.write("- Download Results")

# ---------------- DATASET ----------------

elif page == "Dataset":

    st.title("Dataset")

    st.write(df.head())

    st.write("Rows and Columns")

    st.write(df.shape)

    st.write("Columns")

    st.write(df.columns)
    # ---------------- MANUAL PREDICTION ----------------

elif page == "Manual Prediction":

    st.title("Manual Prediction")

    study_hours = st.number_input("Weekly Self Study Hours", min_value=0.0)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
    assignments = st.number_input("Assignments Completed", min_value=0)
    total_score = st.number_input("Total Score", min_value=0.0, max_value=100.0)

    if st.button("Predict"):

        if total_score < 0 or total_score > 100:
            st.error("⚠️ Please enter a valid Total Score between 0 and 100.")
        else:
                data = np.array([[study_hours, attendance, assignments, total_score]])
                data = scaler.transform(data)
                prediction = model.predict(data)
            
            # Check if total score is between 0 and 25 to show Fail
                if 0 <= total_score <= 25:
                    final_grade = "Fail"
                else:
                    final_grade = prediction
                
                st.success(f"Predicted Grade: {final_grade}")

        # ---------------- CSV UPLOAD ----------------

elif page == "CSV Upload":

        st.title("CSV Batch Prediction")

        uploaded_file = st.file_uploader(
             "Upload zip File",
             type=["zip"]
       )


        if uploaded_file is not None:
            data = pd.read_csv(zipfile.ZipFile(uploaded_file).open("dataset.csv"))
        else:
            data = pd.read_csv(zipfile.ZipFile("dataset.zip").open("dataset.csv"))
            st.info("ℹ️ Automatically running batch prediction using default dataset.zip")

        st.write("### Uploaded Data")
        st.write(data.head())
        
        # Process and drop columns safely
        input_data = data.drop(columns=["student_id", "grade"], errors="ignore")
        input_scaled = scaler.transform(input_data)
        predictions = model.predict(input_scaled)
        
        data["Predicted Grade"] = predictions
        st.write("### Prediction Results")
        st.write(data)
        
        # Create a download button for the predictions
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        # ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("Model Performance")

    st.subheader("Best Model")
    st.success("Random Forest Classifier")

    st.subheader("Accuracy")
    st.write("99.81%")

    st.subheader("Algorithms Used")
    st.write("""
- Logistic Regression
- Decision Tree
- Random Forest
""")

    st.subheader("Features Used")
    st.write("""
- Weekly Self Study Hours
- Attendance
- Assignments Completed
- Total Score
""")


# ---------------- ABOUT ----------------

elif page == "About":

    st.title("About Project")

    st.write("""
### Student Performance Prediction Dashboard

This project predicts student grades using Machine Learning.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

### Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest

### Best Model

Random Forest Classifier
""")
