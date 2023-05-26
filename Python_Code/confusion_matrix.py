import pandas as pd
from sklearn.linear_model import LogisticRegression
import gradio as gr
from PIL import Image
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv("survey_lung_cancer.csv")
# Split the data into features and target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Train the model


# Define the Gradio interface
def predict_lung_cancer(gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain,algorithm):
    # Convert the input to a DataFrame
    input_df = pd.DataFrame({
        "GENDER": [2 if gender == "Male" else 1],
        "AGE": [age],
        "SMOKING": [1 if smoking == True else 0],
        "YELLOW_FINGERS": [1 if yellow_fingers == True else 0],
        "ANXIETY": [1 if anxiety == True else 0],
        "PEER_PRESSURE": [1 if peer_pressure == True else 0],
        "CHRONIC DISEASE": [1 if chronic_disease == True else 0],
        "FATIGUE ": [1 if fatigue == True else 0],
        "ALLERGY ": [1 if allergy == True else 0],
        "WHEEZING": [1 if wheezing == True else 0],
        "ALCOHOL CONSUMING": [1 if alcohol_consuming == True else 0],
        "COUGHING": [1 if coughing == True else 0],
        "SHORTNESS OF BREATH": [1 if shortness_of_breath == True else 0],
        "SWALLOWING DIFFICULTY": [1 if swallowing_difficulty == True else 0],
        "CHEST PAIN": [1 if chest_pain == True else 0],
    })
    algorithm = inputs[-1].value
    rf = RandomForestClassifier()
    if algorithm == "Random Forest":
        rf = RandomForestClassifier()
    elif algorithm == "Support Vector Machine":
        rf = SVC()
    elif algorithm == "Logistic Regression":
        rf = LogisticRegression(max_iter=10000)
    elif algorithm == "K-nearest neighbors":
        rf=KNeighborsClassifier()
    elif algorithm == "Gaussian Naive Bayes":
        rf=GaussianNB()
    elif algorithm == "Decision Tree":
        rf= DecisionTreeClassifier()
    rf.fit(X, y)
    # Make the prediction
    prediction = rf.predict(input_df)[0]

    # Return the prediction
    if len(input_df.columns[(input_df == 0).all()]) == len(input_df.columns) - 2 :
        pre = "please entre your symtoms"
    elif prediction == 0:
        pre = "Low likelihood of lung cancer"
    else :
        pre= "Hight likelihood of lung cancer"
    return pre 
inputs = [
    
    gr.inputs.Radio(["Male", "Female"], label="Gender",default="Male"),
    gr.inputs.Slider(18, 100, label="Age", step=1),
    gr.inputs.Checkbox(label="Smoking"),
    gr.inputs.Checkbox(label="Yellow Fingers"),
    gr.inputs.Checkbox(label="Anxiety"),
    gr.inputs.Checkbox(label="Peer Pressure"),
    gr.inputs.Checkbox(label="Chronic Disease"),
    gr.inputs.Checkbox(label="Fatigue"),
    gr.inputs.Checkbox(label="Allergy"),
    gr.inputs.Checkbox(label="Wheezing"),
    gr.inputs.Checkbox(label="Alcohol Consuming"),
    gr.inputs.Checkbox(label="Coughing"),
    gr.inputs.Checkbox(label="Shortness of Breath"),
    gr.inputs.Checkbox(label="Swallowing Difficulty"),
    gr.inputs.Checkbox(label="Chest Pain"),
    gr.inputs.Dropdown(["Decision Tree","Gaussian Naive Bayes","K-nearest neighbors","Random Forest", "Support Vector Machine", "Logistic Regression"], label="Algorithm",default="Random Forest"),

    
]

outputs = gr.outputs.Label(label="Prediction")


gradio_ui = gr.Interface(fn=predict_lung_cancer, inputs=inputs, outputs=outputs, title="Lung Cancer Prediction",
                         
                         css="footer {visibility: hidden} #warning {background-color: red}",theme=gr.themes.Soft(), live=False)


gradio_ui.launch()