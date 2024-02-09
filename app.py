import pandas as pd
import streamlit as st
import joblib
import numpy as np

model= joblib.load("titaniv_v0.sav")

def main():
    """ main() contains all UI structure elements; getting and storing user data can be done within it"""
    st.title("Titanic Survival Prediction")                                                                             
    st.write("""## Would you have survived From Titanic Disaster?""")                                                   

    st.title("-----          Check Your Survival Chances          -----")

    ## Framing UI Structure
    age = st.slider("Enter Age :", 1, 75, 30)                                                                  

    fare = st.slider("Fare (in 1912 $) :", 15, 500, 40)                                                        

    Parch = st.selectbox("How many Parents or children are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8]) 

    sex = st.selectbox("Select Gender:", ["Male","Female"])                         
    if (sex == "Male"):                                                            
        Sex=0
    else:
        Sex=1

    Pclass= st.selectbox("Select Passenger-Class:",[1,2,3])                        

    Embarked_C,Embarked_Q,Embarked_S=0,0,0                 
    if boarded_location == "Queenstown":
        Embarked_Q=1
    elif boarded_location == "Southampton":
        Embarked_S=1
    else:
        Embarked_C=1

    data={"Age":age,"Fare":fare,"Parch":Parch,"Sex":Sex,"Pclass":Pclass,"Embarked_Q":Embarked_Q,"Embarked_S":Embarked_S,"Embarked_C":Embarked_C}

    df=pd.DataFrame(data,index=[0])     
    return df

data=main()                           

if st.button("Predict"):                                                                
    result = model.predict(data)                                                        
    proba=model.predict_proba(data)                                                    

    if result[0] == 1:
        st.write("***congratulation !!!....*** **You probably would Survive!**")
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2),round((proba[0,1])*100,2)))
    else:
        st.write("***Better Luck Next time !!!!...*** **you're probably Die!'**")
        st.image(r"Rip.jfif")
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2),round((proba[0,1])*100,2)))
