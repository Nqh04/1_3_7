import streamlit as st
import pandas as pd
import joblib # For loading the model and preprocessor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np # Import numpy

# Load the trained classification model, preprocessor, and label encoder
try:
    model_classification = joblib.load('model_classification.joblib')
    preprocessor_classification = joblib.load('preprocessor_classification.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Error: Model, preprocessor, or label encoder not found. Please run the training notebook first.")
    st.stop()

st.title('Product Type Prediction App')

st.sidebar.header('User Input Features')

# Get the categories from the fitted OneHotEncoder
# Assuming the OneHotEncoder is the second transformer in the ColumnTransformer pipeline
onehot_encoder = preprocessor_classification.transformers_[1][1]
categorical_features_used_in_preprocessor = ['Season', 'Gender', 'store_location', 'product_category', 'Size'] # Define the features used in the preprocessor
categorical_categories = [onehot_encoder.categories_[preprocessor_classification.feature_names_in_.tolist().index(feature)] for feature in categorical_features_used_in_preprocessor]


# Create a dictionary to map feature names to their categories
categories_dict = {
    'Season': categorical_categories[categorical_features_used_in_preprocessor.index('Season')],
    'Gender': categorical_categories[categorical_features_used_in_preprocessor.index('Gender')],
    'store_location': categorical_categories[categorical_features_used_in_preprocessor.index('store_location')],
    'product_category': categorical_categories[categorical_features_used_in_preprocessor.index('product_category')],
    'Size': categorical_categories[categorical_features_used_in_preprocessor.index('Size')]
}

def user_input_features():
    season = st.sidebar.selectbox('Season', categories_dict['Season'])
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.sidebar.selectbox('Gender', categories_dict['Gender'])
    store_location = st.sidebar.selectbox('Store Location', categories_dict['store_location'])
    total_bill = st.sidebar.number_input('Total Bill', min_value=0, value=50000)
    product_category = st.sidebar.selectbox('Product Category', categories_dict['product_category'])
    size = st.sidebar.selectbox('Size', categories_dict['Size'])


    data = {'Season': season,
            'Age': age,
            'Gender': gender,
            'store_location': store_location,
            'Total_Bill': total_bill,
            'product_category': product_category,
            'Size': size}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Apply the preprocessor to the user input features
# Ensure the column order matches the order used during training
X_classification_columns = ['Season', 'Age', 'Gender', 'store_location', 'Total_Bill', 'product_category', 'Size']
input_df = input_df[X_classification_columns] # Reindex to ensure correct column order

input_df_processed = preprocessor_classification.transform(input_df)

# Make prediction
prediction_encoded = model_classification.predict(input_df_processed)

# Decode the prediction
prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

st.subheader('Prediction')
st.write(prediction_decoded[0])
