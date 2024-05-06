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
    scaled_data = scaler.fit_transform(df)  # No need to access the 'values' attribute
    return scaled_data, scaler

# Define a function to predict energy consumption for the specified time
def predict_energy_consumption(input_data, scaler, time):
    # Prepare the input data
    prepared_input = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))  # Reshape the input for LSTM
    # Predict energy consumption
    predictions = model.predict(prepared_input)
    
    # Inverse transform the predictions to get the original scale of energy consumption
    predictions = scaler.inverse_transform(predictions)
    
    # Calculate total energy consumption
    total_energy_consumption = np.sum(predictions) * time 
    
    # Calculate cost based on energy consumption (assuming php12 per kWh)
    cost = total_energy_consumption * 12
    
    # Calculate time in hours
    time_hours = time   
    
    return predictions, total_energy_consumption, cost, time_hours

# Streamlit app
def main():
    st.title('Smart Energy Management App')


    st.write('Enter Appliance Data:')
    
    # Input fields for each appliance
    appliance_data = []
    for i in range(3):
        st.write('## Appliance {}:'.format(i+1))
        energy_consumption = st.number_input('Energy Consumption (kWh) for Appliance {}'.format(i+1), min_value=0.0, step=0.01)
        appliance_data.append(energy_consumption)
    
    # Input field for time (in minutes)
    time = st.number_input('Enter Time (in hour)', min_value=0, step=1)
    
    # Prepare input data for prediction
    input_data = np.array([appliance_data])
    scaled_input_data, scaler = prepare_data(input_data)

    # Button to trigger prediction
    if st.button('Predict Energy Consumption'):
        # Predict energy consumption
        predictions, total_energy_consumption, cost, time_hours = predict_energy_consumption(scaled_input_data, scaler, time)

        # Display the predictions
        st.write("## Energy Consumption Prediction:")
        for i, pred in enumerate(predictions[0]):
            st.write("Appliance {}: {} kWh".format(i+1, pred))
        
        st.write("Total Energy Consumption: {:.2f} kWh".format(total_energy_consumption))
        st.write("Time: {:.2f} hours".format(time_hours))
        st.write("Cost: PHP{:.2f}".format(cost))


if __name__ == "__main__":
    main()
