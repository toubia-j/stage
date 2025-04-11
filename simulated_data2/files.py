import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold ,cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error



consommation_partitionner_24h = {
    1: pd.read_csv("consommation_partitionner_24h_1.csv"),
    2: pd.read_csv("consommation_partitionner_24h_2.csv"),
    3: pd.read_csv("consommation_partitionner_24h_3.csv"),
    4: pd.read_csv("consommation_partitionner_24h_4.csv")
}

consommation_partitionner_48h = {
    1: pd.read_csv("consommation_partitionner_48h_1.csv"),
    2: pd.read_csv("consommation_partitionner_48h_2.csv"),
    3: pd.read_csv("consommation_partitionner_48h_3.csv"),
    4: pd.read_csv("consommation_partitionner_48h_4.csv")
}

conso_tou = pd.read_csv("conso_heat_perif_toulouse")
conso_zur = pd.read_csv("conso_heat_perif_zurich")
conso_sev = pd.read_csv("conso_cool_perif_seville")
occ=pd.read_csv("Occupancy_per_hour",delimiter="\t")

files = {
    "agen": "Meteo_Perif_Toulouse_Contemporain/Agen/Simulation_Outputs",
    "albi": "Meteo_Perif_Toulouse_Contemporain/Albi/Simulation_Outputs",
    "auch": "Meteo_Perif_Toulouse_Contemporain/Auch/Simulation_Outputs",
    "toulouse": "Meteo_Perif_Toulouse_Contemporain/Toulouse/Simulation_Outputs",
    "Birmensdorf":"Meteo_Perif_Zurich_Contemporain/Birmensdorf/Simulation_Outputs",
    "Taenikon":"Meteo_Perif_Zurich_Contemporain/Taenikon/Simulation_Outputs",
    "Zurich_fluntern":"Meteo_Perif_Zurich_Contemporain/Zuerich_Fluntern/Simulation_Outputs",
    "Zurich_kloten":"Meteo_Perif_Zurich_Contemporain/Zuerich_kloten/Simulation_Outputs",
    "Cordoba": "Meteo_Perif_Seville_Contemporain/Cordoba/Simulation_Outputs",
    "Granada": "Meteo_Perif_Seville_Contemporain/Granada/Simulation_Outputs",
    "Malaga": "Meteo_Perif_Seville_Contemporain/Malaga/Simulation_Outputs",
    "Sevilla": "Meteo_Perif_Seville_Contemporain/Sevilla/Simulation_Outputs"   
}

files3 = {
    "agen": "Meteo_Perif_Toulouse_Contemporain/Agen/Meteo_input",
    "albi": "Meteo_Perif_Toulouse_Contemporain/Albi/Meteo_input",
    "auch": "Meteo_Perif_Toulouse_Contemporain/Auch/Meteo_input",
    "toulouse": "Meteo_Perif_Toulouse_Contemporain/Toulouse/Meteo_input",
    "Birmensdorf":"Meteo_Perif_Zurich_Contemporain/Birmensdorf/Meteo_input",
    "Taenikon":"Meteo_Perif_Zurich_Contemporain/Taenikon/Meteo_input",
    "Zurich_fluntern":"Meteo_Perif_Zurich_Contemporain/Zuerich_Fluntern/Meteo_input",
    "Zurich_kloten":"Meteo_Perif_Zurich_Contemporain/Zuerich_kloten/Meteo_input",
    "Cordoba": "Meteo_Perif_Seville_Contemporain/Cordoba/Meteo_input",
    "Granada": "Meteo_Perif_Seville_Contemporain/Granada/Meteo_input",
    "Malaga": "Meteo_Perif_Seville_Contemporain/Malaga/Meteo_input",
    "Sevilla": "Meteo_Perif_Seville_Contemporain/Sevilla/Meteo_input"  
}



models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVC": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

models2 = {
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Logistic Regression": MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=1000)),
    "SVC": MultiOutputClassifier(SVC(random_state=42)),
    "KNN": MultiOutputClassifier(KNeighborsClassifier()),
    "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
}