import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

df = pd.read_csv("avocado_kaggle.csv")

# date to time so we can use it
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.reset_index(drop=True)

# Encode categorical variables (type and region), because we will use it for the prediction
df_encoded = pd.get_dummies(df[['type', 'region']], drop_first=True)

# Concatenate encoded features with the original dataset to use it for the prediction
df_final = pd.concat([df[['AveragePrice']], df_encoded], axis=1)

# Split the data into features (X) and target variable (y)
X = df_final.drop('AveragePrice', axis=1)
y = df_final['AveragePrice']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Create a Streamlit app
st.title('Avocado Price Calculator🥑')

# create a box for avocado type
avocado_type = st.selectbox('Select Avocado Type', df['type'].unique())

# create a box or region
region = st.selectbox('Select Region', df['region'].unique())

# Create a button to calculate the price
if st.button('Calculate Price of your little avocado 💗🔍'): # this is going to be the name of the button
    # Encode the selected avocado type and region
    type_encoded = pd.get_dummies(pd.Series([avocado_type]).astype('category'), drop_first=True)
    region_encoded = pd.get_dummies(pd.Series([region]).astype('category'), drop_first=True)

    # Create a new DataFrame with all columns from training data
    prediction_data = pd.DataFrame(0, columns=X.columns, index=[0])

    # Update corresponding columns with encoded values
    prediction_data.update(type_encoded)
    prediction_data.update(region_encoded)

    # Predict the avocado price
    predicted_price = model.predict(prediction_data)
    st.success(f'The predicted price of {avocado_type} avocado in {region} region is ${predicted_price[0]:.2f}')
