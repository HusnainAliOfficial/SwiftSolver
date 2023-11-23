import streamlit as st
import pickle

model=''

st.title("Sales Perdiction")

selected_option = st.selectbox("Select an model you want to Load ", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
st.write(f"You selected: {selected_option}")

if selected_option=="Linear Regression" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/5_sales_perdiction/linear.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Decision Tree Regressor" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/5_sales_perdiction/DecisionTree.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Random Forest Regressor" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/5_sales_perdiction/RandomForest.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
         
         
         
value1 = st.number_input("Enter Value of TV",format="%.6f")
value2 = st.number_input("Enter Value of Radio",format="%.6f")
value3 = st.number_input("Enter Value of Newspaper",format="%.6f")



if st.button("Predict"):
        input_values =[float(value1),float(value2), float(value3)]
        prediction = model.predict([input_values])[0] # type: ignore
        st.write(f"Selling Price : {prediction}")
    
 

