import streamlit as st
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
# Load and preprocess data
data = pd.read_csv("Student grade dataset2.csv")  # Update with your dataset filename
with open("deepak_model_params.json",'r') as json_file:
    model_params=json.load(json_file)
    
scaler=StandardScaler()
model=xgb.XGBRegressor(**model_params)
model.load_model("Trained_model.json")
# Streamlit app
def main():
    state=1
    idno=1
    st.title("Public Exam Score Prediction")
    # Input fields for user to enter scores
    st.write("## Enter Scores to Predict Public Exam Score (%)")
    other_scores = {}
    for col in data.columns:
        if col not in ['State','Final grade','Name ','Id no.']:
            other_scores[col] = st.number_input(f"{col}", min_value=0, max_value=100)
    
    if st.button("Predict"):
        input_data = [other_scores[col] for col in data.columns if col not in ['State','Final grade','Name ','Id no.']]
        input_data = scaler.fit_transform([input_data])
        prediction = model.predict(input_data)[0]
        st.write(f"Predicted Public Exam Score (%): {prediction}")

if __name__ == "__main__":
    main()
