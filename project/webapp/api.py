import streamlit as st
import requests
import time

import pandas as pd
import pickle
import io
import os
from PIL import Image
import base64


def set_local_image_with_opacity(image_path, opacity):
    image = open(image_path, 'rb').read()
    image_base64 = base64.b64encode(image).decode()
    background_css = f'''
    <style>
    .stApp {{
        background-image: url('data:image/png;base64,{image_base64}');
        background-size: cover;
        opacity: {opacity};
    }}
    </style>
    '''
    st.markdown(background_css, unsafe_allow_html=True)
set_local_image_with_opacity('logo.png', 0.9)


st.title('Scoring Interface')

extension_attendue =[".csv",".wav"]

def verifier_extension_fichier(uploaded_file, extensions_attendues):
    nom_fichier = uploaded_file.name
    extension_trouvee = None
    for ext in extensions_attendues:
        if nom_fichier.lower().endswith(ext.lower()):
            extension_trouvee = ext
            break
    return extension_trouvee, extension_trouvee is not None

def predict_csv(uploaded_file):
    file = {'file': ("uploaded_file.csv", uploaded_file.getvalue(), 'csv')}
    serving_api_url = "http://serving-api:8080/predict"
    response = requests.post(serving_api_url, files=file, auth=('user', 'pass'))
    if response.status_code == 200:
        return response.json()
    else:
        return None

def predict_audio(uploaded_file):
    file = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'wav')}
    serving_api_url = "http://serving-api:8080/predictaudio"
    response = requests.post(serving_api_url, files=file, auth=('user', 'pass'))
    if response.status_code == 200:
        return response.json()
    else:
        return None

def save_feedback(feedback):
    # Logique pour enregistrer le feedback
    pass

uploaded_file = st.file_uploader("Importer un fichier CSV ou WAV")

if uploaded_file is not None:
    extension, est_valide = verifier_extension_fichier(uploaded_file, extension_attendue)

    if est_valide:
        if extension == ".csv":
            if st.button("Prédire"):
                prediction = predict_csv(uploaded_file)
                if prediction is not None:
                    for pred in prediction:
                        result = "bon payeur" if pred == 1 else "mauvais payeur"
                        st.write("Voici la prédiction :", result)
                    st.session_state.prediction = prediction
                    st.session_state.show_feedback_button = True
                else:
                    st.write("Erreur lors de la prédiction.")
        
        elif extension == ".wav":
            if st.button("Prédire audio"):
                prediction = predict_audio(uploaded_file)
                if prediction is not None:
                    for pred in prediction:
                        result = "sans danger" if pred == 0 else "danger mort imminente"
                        st.write("Voici la prédiction :", result)
                    st.session_state.prediction = prediction
                    st.session_state.show_feedback_button = True
                else:
                    st.write("Erreur lors de la prédiction.")

        if st.session_state.get('show_feedback_button', False):
            feedbackchoice = st.radio("Feedback :", options=["Incorrect", "Correct"])
            feedback_button_clicked = st.button("Soumettre le feedback")
            if feedback_button_clicked:
                 # Si le feedback est "Incorrect", lancer l'API de feedback dans une autre API
                if feedbackchoice == "Incorrect":
                    feedback_url = "http://serving-api:8080/feedback"
                    response = requests.get(feedback_url, auth=('user', 'pass'))
                    if response.status_code == 200:
                        print("API de feedback lancée avec succès.")
                    else:
                        print(f"Échec de la requête vers l'API de feedback. Code d'erreur : {response.status_code}")
                st.write("Feedback enregistré.")
                st.session_state.show_feedback_button = False
                # Attendre le délai spécifié
                time.sleep(1) #1 seconde
                # Recharger la page
                st.rerun()
                
    else:
        st.write("Le fichier doit avoir l'une des extensions suivantes :", ", ".join(extension_attendue))
else:
    st.write("Aucun fichier n'a été importé")

#le fichier data-prod fermé pour qu'on puisse avoir les output