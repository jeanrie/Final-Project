import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Define a function to prepare data for prediction
def prepare_data(df):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)  
    return scaled_data, scaler

# Define a function to predict energy consumption for the specified time
def predict_energy_consumption(input_data, scaler, time):

    prepared_input = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))  

    # Predict energy consumption
    predictions = model.predict(prepared_input)
    

    predictions = scaler.inverse_transform(predictions)
    
    total_energy_consumption = np.sum(predictions)
    
    cost_php = total_energy_consumption * 12
    
    return predictions, total_energy_consumption, cost_php

# Streamlit app
def main():
    st.title('Energy Consumption Prediction')

    st.write('Enter Appliance Data:')
    
    appliance_data = []
    for i in range(3):
        st.write('## Appliance {}:'.format(i+1))
        energy_consumption = st.number_input('Energy Consumption (kWh) for Appliance {}'.format(i+1), min_value=0.0, step=0.01)
        appliance_data.append(energy_consumption)
    
    # Input field for time (in hours)
    time = st.number_input('Enter Time (in hours)', min_value=0.0, step=0.01)
    
    # Prepare input data for prediction
    input_data = np.array([appliance_data])
    scaled_input_data, scaler = prepare_data(input_data)

    # Button to trigger prediction
    if st.button('Predict Energy Consumption'):
        # Predict energy consumption
        predictions, total_energy_consumption, cost_php = predict_energy_consumption(scaled_input_data, scaler, time)

        # Display the predictions
        st.write("## Energy Consumption Prediction:")
        for i, pred in enumerate(predictions[0]):
            st.write("Appliance {}: {} kWh".format(i+1, pred))
        
        st.write("Total Energy Consumption: {:.2f} kWh".format(total_energy_consumption))
        st.write("Cost: PHP {:.2f}".format(cost_php))

if __name__ == "__main__":
    main()
