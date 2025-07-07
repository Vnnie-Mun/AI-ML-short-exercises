# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:10:25 2025

@author: pc
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model correctly
loaded_model = pickle.load(open("C:/Users/pc/Downloads/testing deployment/trained_model.sav", 'rb'))

# Create a function for prediction
def diabetes_prediction(input_data):
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main Streamlit app
def main():
    st.title('Diabetes Prediction Web App')

    # Getting input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Prediction
    diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            # Convert input values to correct types
            input_data = [
                int(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
            ]
            diagnosis = diabetes_prediction(input_data)
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

if __name__ == '__main__':
    main()