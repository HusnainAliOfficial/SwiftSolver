import streamlit as st
import pickle

model=""
st.title("IRIS Flower Classification")

selected_option = st.selectbox("Select an model you want to Load ", ["Logestic Regression", "Decision Tree Classifier", "Random Forest Classifier"])
st.write(f"You selected: {selected_option}")

if selected_option=="Logestic Regression" :
    with open('REMOTE_INTERNSHIP/Swift_Solver/1_iris/logestic.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Decision Tree Classifier" :
    with open('REMOTE_INTERNSHIP/Swift_Solver/1_iris/DecisionTree.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
elif selected_option=="Random Forest Classifier" :
    with open('REMOTE_INTERNSHIP/Swift_Solver/1_iris/RandomForest.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
         
         
         
         
value1 = st.number_input("Enter Sepal Length",step=0.000001)
value2 = st.number_input("Enter Sepal Width",step=0.000001)
value3 = st.number_input("Enter Petal Length",step=0.000001)
value4 = st.number_input("Enter Petal Width",step=0.000001)

if st.button("Predict"):
    input_values = [value1, value2, value3, value4]
    prediction = model.predict([input_values])[0] # type: ignore
    st.write(f"Model Prediction: {prediction}")

