from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from csv import writer
import pandas as pd
import numpy as np
import pickle
import librosa
import io

clfs={
    'CART': DecisionTreeClassifier(criterion='gini',random_state=1),
    'ID3' : DecisionTreeClassifier(criterion='entropy', random_state=1),
    'Stump' : DecisionTreeClassifier(criterion='gini',max_depth=1, random_state=1),
    'KNN' : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Bag' : BaggingClassifier(estimator=MLPClassifier(hidden_layer_sizes=(10),max_iter=100,learning_rate_init=0.001,random_state=1),n_estimators=100,random_state=1,n_jobs=-1),
    'Ad': AdaBoostClassifier(n_estimators=200,random_state=1),
    'RF' : RandomForestClassifier(n_estimators=200,random_state=1,n_jobs=-1),
    'ExtraTree' : ExtraTreesClassifier(n_estimators=200,random_state=1,n_jobs=-1)
}


def preparation (churn):
    #Suppression de la colonne
    churn.drop(['RowNumber','CustomerId'], axis=1,inplace=True)
    X=churn.iloc[:,:-1].values
    Y=churn.iloc[:,-1].values
    num_cols=[0,3,4,5,6,7,8,9]
    X_num=X[:,num_cols]
    SS=StandardScaler()
    X_num_norm=SS.fit_transform(X_num)

    X_gender=X[:,2]
    X_gender[X_gender=='Female']=0
    X_gender[X_gender=='Male']=1
    X_gender=X_gender.astype(int).reshape((-1,1))

    X_geography = X[:,1]
    encoder = OneHotEncoder()
    X_geography = encoder.fit_transform(X_geography.reshape((-1,1))).toarray().astype(int)

    X_final=np.concatenate((X_num,X_gender,X_geography),axis=1)
    X_final_norm=np.concatenate((X_num_norm,X_gender,X_geography),axis=1)


    colonnes=np.concatenate((churn.columns[num_cols].values,['Gender'],np.asanyarray(encoder.categories_).flatten()))
    with open('../artifacts/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    with open('../artifacts/scaler.pkl', 'wb') as f:   
        pickle.dump(SS, f)

    return X_final,X_final_norm,Y,colonnes

'''def preparation_scoring (churn,SS,encoder):
    #Suppression de la colonne
    churn.drop(['RowNumber','CustomerId'], axis=1,inplace=True)
    X=churn.values
    
    num_cols=[0,3,4,5,6,7,8,9]
    X_num=X[:,num_cols]
    X_num_norm=SS.transform(X_num)

    X_gender=X[:,2]
    X_gender[X_gender=='Female']=0
    X_gender[X_gender=='Male']=1
    X_gender=X_gender.astype(int).reshape((-1,1))

    X_geography = X[:,1]
    X_geography = encoder.transform(X_geography.reshape((-1,1))).toarray().astype(int)

    X_final=np.concatenate((X_num,X_gender,X_geography),axis=1)
    X_final_norm=np.concatenate((X_num_norm,X_gender,X_geography),axis=1)

    return X_final,X_final_norm'''
def preparation_scoring(df, SS, encoder, is_audio=False):
    if not is_audio:
        num_cols = [0, 3, 4, 5, 6, 7, 8, 9]
        X_num = df.iloc[:, num_cols].values
        X_num_norm = SS.transform(X_num)
        X_geography = df.iloc[:, 1]
        X_geography = encoder.transform(X_geography.values.reshape((-1, 1))).toarray().astype(int)
        X_final = np.concatenate((X_num, X_geography), axis=1)
        X_final_norm = np.concatenate((X_num_norm, X_geography), axis=1)
    else:
        # Traitement spécifique pour les fichiers audio
        # Exclure la première colonne qui contient probablement le nom du fichier audio
        audio_data = df.iloc[:, 1:].values
        # Réaliser le traitement nécessaire pour les fichiers audio (par exemple : extraction de caractéristiques)
        
        return audio_data
    
    return X_final, X_final_norm


def run_classifiers_cv(X,Y,clfs):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    max_score = 0
    for i in tqdm(clfs):
        clf = clfs[i]
        cv_f1_score = cross_val_score(clf, X, Y, cv=kf, scoring='f1')
        print("F1 Score moyen pour {0} : {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_f1_score), np.std(cv_f1_score)))
        if np.mean(cv_f1_score) >= max_score:
            max_score = np.mean(cv_f1_score)
            best_model = clf
    return best_model,max_score

def comparaison_classifieurs(X,Y,X_norm,clfs):
    meilleur_model,meilleur_score=run_classifiers_cv(X,Y,clfs)
    meilleur_model_norm,meilleur_score_norm=run_classifiers_cv(X_norm,Y,clfs)
    if (meilleur_score>=meilleur_score_norm):
        strategie='no_norm'
        with open('../artifacts/strategie.pkl', 'wb') as f:   
            pickle.dump(strategie, f)
        return meilleur_model,strategie
    else:
        strategie='norm'
        with open('../artifacts/strategie.pkl', 'wb') as f:   
            pickle.dump(strategie, f)
        return meilleur_model_norm,strategie
    
def feature_selection(X,X_norm,Y,colonnes, best_model,strategie):
    RF = RandomForestClassifier(n_estimators=1000,random_state=1)
    RF.fit(X,Y)
    importances=RF.feature_importances_

    std=np.std([tree.feature_importances_ for tree in RF.estimators_],axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    print(colonnes[sorted_idx])
    padding = np.arange(X.size / len(X)) + 0.5

    if(strategie=='no_norm'):
        X1=X.copy()
    else:
        X1=X_norm.copy()

    scores=np.zeros(X1.shape[1])
    kf=KFold(n_splits=10, shuffle=True, random_state=0)
    for f in np.arange(0,X1.shape[1]):
        cv_f1_score=cross_val_score(best_model,X1[:,sorted_idx[:f+1]], Y, cv=kf, scoring=('f1'))
        scores[f]=np.mean(cv_f1_score)

    selected_features=sorted_idx[:np.argmax(scores)+1]
    return selected_features


def select_parameters(X,X_norm,Y,best_model,strategie,selected_features):

    be1=DecisionTreeClassifier(max_depth=1,random_state=1)
    be2=DecisionTreeClassifier(max_depth=3,random_state=1)
    be3=DecisionTreeClassifier(max_depth=4,random_state=1)
    be4=DecisionTreeClassifier(max_depth=5,random_state=1)


    parametres_adaboost = {'base_estimator':(be1,be2,be3,be4),
                        'n_estimators' : (100,200,500,1000)}

    parametres_bagging = {'base_estimator':(be1,be2,be3,be4),
                        'n_estimators' : (100,200,500,1000)}

    parametres_ExtraTree = {'n_estimators' : (100,200,500)}

    parametres_rf = {'n_estimators' : (100,200,500)}

    parametres_dt = {'criterion' : ('gini','entropy'), 'max_depth' :(1,2,3,4,5)}

    parametres_knn = {'n_neighbors' : (1,3,5,7,9,10)}

    parametres_mlp = {'hidden_layer_sizes' : ((40,20),(30,10),(20,10),(20),(10)), 'activation' : ('relu','tanh', 'logistic')}

    if(type(best_model)==BaggingClassifier):
        clf=BaggingClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_bagging

    elif(type(best_model)==RandomForestClassifier):
        clf=RandomForestClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_rf

    elif(type(best_model)==ExtraTreesClassifier):
        clf=ExtraTreesClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_ExtraTree

    elif(type(best_model)==AdaBoostClassifier):
        clf=AdaBoostClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_adaboost

    elif(type(best_model)==KNeighborsClassifier):
        clf=KNeighborsClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_knn

    elif(type(best_model)==DecisionTreeClassifier):
        clf=DecisionTreeClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_dt

    else:
        clf=MLPClassifier(random_state=1,n_jobs=-1)
        parameters=parametres_mlp

    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1')

    if strategie =='no_norm':
        X_selected = X[:,selected_features]
    else:
        X_selected = X_norm[:,selected_features]

    GS.fit(X_selected,Y)

    print(GS.best_score_)
    return GS.best_estimator_

def automatisation (X,X_norm,Y,strategie,classifieur,selected_features):
    RF = RandomForestClassifier(n_estimators=1000,random_state=1)

    P = Pipeline([('FS',SelectFromModel(RF,max_features=len(selected_features))), ('classifieur',classifieur)])

    if strategie=='no_norm':
        P.fit(X,Y)
    else:
        P.fit(X_norm,Y)
    with open('../artifacts/model_final.pkl', 'wb') as f:   
        pickle.dump(P, f)


def prepa_audio(lien, taux):
    data, _ = librosa.load(lien, sr=taux)
    df = pd.DataFrame(data.reshape(1, -1))
    return df

