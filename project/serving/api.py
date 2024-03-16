from fastapi import FastAPI, File, UploadFile, HTTPException
from utils import *
from utils import preparation_scoring

app = FastAPI()

# Load model
with open("../artifacts/model_final.pkl", 'rb') as f:
    model = pickle.load(f)
with open("../artifacts/strategie.pkl", 'rb') as f:
    strategie = pickle.load(f)
with open("../artifacts/encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)
with open("../artifacts/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("Before reading CSV file")
        df = pd.read_csv(file.file, sep=';', header=None)
        print("After reading CSV file")

        print(df[0].dtype)
        print(df)
        X_final_scoring,X_final_norm_scoring=preparation_scoring(df,scaler,encoder)

        try:
            if(strategie=='no_norm'):
                predictions=model.predict(X_final_scoring)
            else:
                predictions=model.predict(X_final_norm_scoring)
        except Exception as e:
            raise HTTPException(status_code=501, detail=str(e))

        # Format predictions as JSON
        predictions_json =  predictions.tolist()

        return predictions_json

    except Exception as e:
        print("Error in predict function:", e)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predictaudio")
async def predict(file: UploadFile = File(...)):
    try:
        print("debut du read")
        with open("uploaded_file.wav", "wb") as f:
            f.write(await file.read())
        print("fin du read")
        
        with open("../artifacts/model_final_audio.pkl", 'rb') as f:
            model_final_audio = pickle.load(f)

        to_predict = prepa_audio('uploaded_file.wav', 50)
        prediction = model_final_audio.predict(to_predict)

        predictions_json =  prediction.tolist()

        list_data=[file.filename]+to_predict.values.tolist()[0]+[prediction[0],prediction[0]]

        with open("../data/prod-data.csv", 'a',newline= '') as f:
            writer_object = writer(f,delimiter=';')
            writer_object.writerow(list_data)

        return predictions_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback")
async def feedback():
    try:
        # ouvrir le fichier prod_data.csv, modifier la dernière colonne de la dernière ligne
        # si la valeur est 1, la mettre à 0, sinon la mettre à 1
        data = pd.read_csv("../data/prod-data.csv", sep=";", header=None)
        print("x")
        if data.iloc[-1, -1] == 0:
            data.iloc[-1, -1] = 1
        elif data.iloc[-1, -1] == 1:
            data.iloc[-1, -1] = 0
        data.to_csv("../data/prod-data.csv", index=False, sep=";", header=False)

        return 200
    except Exception as e:
        print(f"Erreur lors de la requête '/feedback': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la requête '/feedback': {str(e)}")
