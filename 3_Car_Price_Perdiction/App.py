import streamlit as st
import pickle

model=''

st.title("Car Selling Price Perdiction")

selected_option = st.selectbox("Select an model you want to Load ", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
st.write(f"You selected: {selected_option}")

if selected_option=="Linear Regression" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/3_car_price_perdiction/logestic.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Decision Tree Regressor" :
    with open('//home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/3_car_price_perdiction/DecisionTree.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Random Forest Regressor" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/3_car_price_perdiction/RandomForest.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
         
         
         
value1 = st.number_input("Enter Car Name Number", step=1)
value2 = st.number_input("Enter Year of Manufaturing After Normalize",format="%.6f")
value3 = st.number_input("Enter Present Price",format="%.6f")
value4 = st.number_input("Enter Driven_KMS",format="%.6f")
value5 = st.number_input("Enter Fuel Type", step=1)
value6 = st.number_input("Enter Selling Type", step=1)
value7 = st.number_input("Enter Transmission", step=1)
value8 = st.number_input("Enter Owner", step=1)


if st.button("Predict"):
        input_values =[int(value1),float(value2), float(value3), float(value4),int(value5), int(value6),int(value7),float(value8)]
        prediction = model.predict([input_values])[0] # type: ignore
        st.write(f"Selling Price : {prediction}")
    
 

