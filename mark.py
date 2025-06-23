import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("gbc_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load saved LabelEncoders
with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

# Streamlit UI
st.title("Gender Prediction")

# User input fields
branch = st.text_input("Branch")
city = st.text_input("City")
customer_type = st.text_input("Customer Type")
product_line = st.text_input("Product Line")
unit_price = st.number_input("Unit Price", min_value=0.0, step=0.1)
quantity = st.number_input("Quantity", min_value=1, step=1)
tax_5_percent = st.number_input("Tax 5%", min_value=0.0, step=0.01)
total = st.number_input("Total", min_value=0.0, step=0.01)
payment = st.text_input("Payment")
cogs = st.number_input("COGS", min_value=0.0, step=0.01)
gross_margin_percentage = st.number_input("Gross Margin Percentage", min_value=0.0, step=0.01)
gross_income = st.number_input("Gross Income", min_value=0.0, step=0.01)
rating = st.number_input("Rating", min_value=0.0, max_value=10.0, step=0.1)

# Convert categorical inputs to numerical using LabelEncoders
def encode_input(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # Default value for unseen categories

if st.button("Predict Gender"):
    input_data = pd.DataFrame([[branch, city, customer_type, product_line, unit_price, quantity,
                                tax_5_percent, total, payment, cogs,
                                gross_margin_percentage, gross_income, rating]],
                              columns=["Branch", "City", "Customer type", "Product line", 
                                       "Unit price", "Quantity", "Tax 5%", "Total", 
                                       "Payment", "cogs", "gross margin percentage", 
                                       "gross income", "Rating"])
    
    # Encode categorical inputs
    for col in ["Branch", "City", "Customer type", "Product line", "Payment"]:
        input_data[col] = encode_input(input_data[col][0], encoders[col])

    # Select only features that were used during training
    input_data = input_data[model.feature_names_in_]

    # Predict the gender
    prediction = model.predict(input_data)[0]
    gender = encoders["Gender"].inverse_transform([prediction])[0]
    
    st.success(f"Predicted Gender: {gender}")
