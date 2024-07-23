import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('trained_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


def make_prediction(model, input_data):
    # Reshape the input data for prediction
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)

    # Make predictions
    prediction = model.predict(input_data_reshaped)

    return prediction


def main():
    st.title('Malicious vs. Not Malicious Prediction App')

    # Get user input
    feature1 = st.slider('Feature 1', 0, 10, 1)
    feature2 = st.slider('Feature 2', 0, 1000000, 4777)
    # Add sliders for other features as needed

    # Create a button to make predictions
    if st.button('Make Prediction'):
        # Make prediction based on user input
        input_data = (feature1, feature2, 5092282,10,711000000,1.071e+10,3,1790,0,0,0,0,1,3753,5227812,0)
        prediction = make_prediction(loaded_model, input_data)

        # Display the prediction result
        st.write('Prediction Result:')
        if prediction[0] == 0:
            st.write('Not malicious')
        else:
            st.write('Malicious')


if __name__ == '__main__':
    main()
    #45304
    #48294064, 100, 716000000, 1.010000e+11, 3, 1943, 13535, 14428310, 451, 0, 3, 143928631, 3917, 0