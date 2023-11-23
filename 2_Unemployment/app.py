import streamlit as st
import pickle

model=''

st.title("Unemployment")

selected_option = st.selectbox("Select an model you want to Load ", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
st.write(f"You selected: {selected_option}")

if selected_option=="Linear Regression" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/2_unemployment/logestic.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Decision Tree Regressor" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/2_unemployment/DecisionTree.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Random Forest Regressor" :
    with open('/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/2_unemployment/RandomForest.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
         
         
         
value1 = st.number_input("Enter Region", step=1)
value2 = st.number_input("Enter Estimated Employed",format="%.8f")
value3 = st.number_input("Enter Estimated Labor Participation Rate",format="%.8f")
value7 = st.number_input("Enter Area Rural", step=1)
value4 = st.number_input("Enter Day", step=1)
value5 = st.number_input("Enter month", step=1)
value6 = st.number_input("Enter year", step=1)


if st.button("Predict"):
    if selected_option=="Linear Regression":
        input_values =[int(value1), value2, value3, int(value7),int(value4), int(value5),int(value6)]
        prediction = model.predict([input_values])[0]
        st.write(f"Unemployment Rate : {prediction*100}")
    
    else:
        input_values = [int(value1), value2, value3, int(value7),int(value4), int(value5),int(value6)]
        prediction = model.predict([input_values])[0]
        st.write(f"Unemployment Rate : {prediction*100}")

