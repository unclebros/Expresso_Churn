import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

# Load label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Load target label encoder
target_label_encoder = joblib.load('target_label_encoder.pkl')

# Function to make predictions
def predict(input_data):
    for column, le in label_encoders.items():
        input_data[column] = le.transform([input_data[column]])[0]
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    # Transform the prediction back to the original label
    return target_label_encoder.inverse_transform([prediction[0]])[0]

# Streamlit app layout
st.title('Expresso Churn Prediction')

# Create input fields for each feature
input_data = {}
for column in model.feature_names_in_:
    input_data[column] = st.text_input(f'Enter {column}')

if st.button('Predict'):
    try:
        # Ensure all inputs are provided
        if all(value for value in input_data.values()):
            prediction = predict(input_data)
            st.write(f'The predicted churn value is: {prediction}')
        else:
            st.write('Please provide all inputs')
    except ValueError as e:
        st.write(f'Error in prediction: {e}')
