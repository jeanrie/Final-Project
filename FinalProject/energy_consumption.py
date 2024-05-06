import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the LSTM model
model = load_model('vanilla_lstm.h5')

# Function to preprocess input data
def preprocess_input_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    return scaled_data, scaler

# Function to make predictions


def predict_energy_consumption(model, input_data):
    # Reshape input data to match model's input shape
    input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return prediction


# Streamlit app
def main():
    st.title("Energy Consumption Prediction")

    # User input for appliance data
    st.sidebar.subheader("Enter Appliance Data:")
    appliance_data = {}
    for i in range(3):  # Assuming 3 appliances
        st.sidebar.write("Appliance {}: ".format(i+1))
        energy_consumption = st.sidebar.number_input('Energy Consumption (kWh) for Appliance {}:'.format(i+1), min_value=0.0, step=0.01)
        appliance_data['Appliance {}'.format(i+1)] = energy_consumption
    
    # Preprocess input data
    input_df = pd.DataFrame(appliance_data, index=[0])
    input_data, scaler = preprocess_input_data(input_df)
    
    # Button to predict energy consumption
    if st.sidebar.button("Predict Energy Consumption"):
        # Make prediction
        prediction = predict_energy_consumption(model, input_data)
        
        # Display prediction
        st.subheader("Predicted Energy Consumption for the next 7 days:")
        for i, pred in enumerate(prediction[0]):
            st.write("Day {}: {} kWh".format(i+1, pred))

if __name__ == "__main__":
    main()
