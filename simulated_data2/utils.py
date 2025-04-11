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
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout,LSTM
import tensorflow as tf

def extract_columns(filepath, column_index=4):
    df = pd.read_csv(filepath, delimiter="\t")
    values = df.iloc[:, column_index].values
    return pd.DataFrame(values.reshape(-1, 24))


def calculate_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

def draw(data, text, X_label,Y_label):   
    plt.figure(figsize=(10, 6))
    for index, row in data.iterrows():
        plt.plot(range(0, 24), row, label=f"Jour {index}")
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(text)
    plt.tight_layout()
    plt.show()


def add_binary_column(df, column_name="heat_on"):
    df[column_name] = (df.drop(columns=[column_name], errors='ignore').sum(axis=1) > 0).astype(int)
    return df


def apply_kmeans(n_clusters,data):
    kmeans=KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans 



def apply_kmeans_dtw(n_clusters, data):
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    kmeans.fit(data)
    return kmeans





def plot_clusters(consommation):  
    min_val = consommation.iloc[:, :-1].min().min() 
    max_val = consommation.iloc[:, :-1].max().max()
    ylim = [min_val - 2, max_val + 2]  
    unique_clusters = consommation["clusters"].unique()  
    num_clusters = len(unique_clusters) 
    num_points = consommation.shape[1] - 1  
    fig, axes = plt.subplots((num_clusters + 1) // 2, 2, figsize=(10, 10)) 
    axes = axes.flatten()
    for i, cluster in enumerate(unique_clusters): 
        ax = axes[i]  
        cluster_data = consommation[consommation["clusters"] == cluster]      
        for index, row in cluster_data.iterrows():
            ax.plot(range(num_points), row.iloc[:-1], color='gray', alpha=0.5)   
        center = cluster_data.iloc[:, :-1].mean(axis=0) 
        ax.plot(range(num_points), center, color='red', label=f'Cluster {cluster} ({len(cluster_data)})')
        ax.set_xlim([0, num_points])  
        ax.set_ylim(ylim) 
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("Heures")
        ax.set_ylabel("Consommation (kJ/h)") 
        ax.legend()
    plt.tight_layout()
    plt.show()



def plot_cluster_centers_with_colors(data_normalized, cluster_assignments, y_label="Valeur"):  
    cluster_centers = []
    for cluster_id in np.unique(cluster_assignments):
        cluster_data = data_normalized[cluster_assignments == cluster_id]
        cluster_center = cluster_data.iloc[:, :-1].mean().values 
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    num_clusters = len(cluster_centers)
    colors = plt.cm.get_cmap('tab10', num_clusters)  
    plt.figure(figsize=(10, 6))
    for cluster_id, cluster_center in enumerate(cluster_centers):
        plt.plot(range(24), cluster_center, color=colors(cluster_id), linewidth=3, label=f"Centre du cluster {cluster_id}")
    plt.title("Centres des Clusters")
    plt.xlabel("Heures")
    plt.ylabel(y_label)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def evaluate_clustering_cooling(consommation_fr,binaire):
    labels = consommation_fr['clusters']
    consommation3_fr = consommation_fr.drop(columns=["clusters", binaire])
    sil_score = silhouette_score(consommation3_fr, labels, metric='euclidean')
    db_score = davies_bouldin_score(consommation3_fr, labels)
    print(f"Davies-Bouldin Index: {db_score}")
    print(f"Silhouette Score: {sil_score}")



def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

"""
Équilibre les clusters en ajustant le cluster majoritaire  pour avoir le même nombre d'exemples 
que le cluster le plus grand parmi les autres.

"""

def balance_clusters(X, y):
    df = X.copy()
    df['clusters'] = y
    cluster_groups = df.groupby('clusters')
    cluster_counts = df['clusters'].value_counts()
    max_other_clusters = cluster_counts[cluster_counts.index != 3].max()
    cluster_3 = cluster_groups.get_group(3)
    cluster_3_resampled = cluster_3.sample(n=max_other_clusters, random_state=42)
    balanced_df = pd.concat([cluster_groups.get_group(cluster) for cluster in cluster_counts.index if cluster != 3] + [cluster_3_resampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    X_balanced = balanced_df.drop(columns=["clusters"])
    y_balanced = balanced_df["clusters"]
    return X_balanced, y_balanced

    

"""
Évalue plusieurs modèles en utilisant un train_test_split. Pour chaque modèle, 
cette fonction calcule la précision (accuracy), le F1-score et le temps d'exécution.
Elle affiche également la matrice de confusion pour chaque modèle.
"""

def evaluate_models_split(X_train, X_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        print(f"\n Évaluation de {name} avec train_test_split...")
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        execution_time = time.time() - start_time
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred) 
        results[name] = {
            "f1_score": f1, 
            "accuracy": accuracy, 
            "execution_time (s)": execution_time
        }
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies classes')
        plt.title(f'Matrice de confusion - {name}')
        plt.show()
        print(f"{name} - Accuracy: {accuracy:.4f} - F1 Score: {f1:.4f} - Temps d'exécution: {execution_time:.4f} sec")
        print("###################################################################")
    return results

"""
Évalue plusieurs modèles en utilisant la validation croisée (cross-validation).
Pour chaque modèle, cette fonction calcule la précision (accuracy) et le F1-score 
avec un nombre défini de plis (folds). Elle calcule également le temps d'exécution.
"""

def evaluate_models_cv(X, y, models, cv=4):
    results = {}
    for name, model in models.items():
        print(f"\n Évaluation de {name} avec Cross Validation ({cv}-folds)...")  
        start_time = time.time()
        scoring = ["accuracy", "f1_weighted"]
        scores = {}
        for metric in scoring:
            score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), scoring=metric)
            scores[metric] = np.mean(score)
        execution_time = time.time() - start_time
        results[name] = scores
        results[name]["execution_time (s)"] = execution_time
        print(f"{name} - Accuracy: {scores['accuracy']:.4f} - F1 Score: {scores['f1_weighted']:.4f} - Temps d'exécution: {execution_time:.4f} sec")
        print("###################################################################")
    return results



############   multilabel

"""
Applique un clustering K-means sur des sous-ensembles de données (6h)
"""

def cluster_data(df, heat_on_column, n_clusters_list, cols_list):
    clustered_dfs = {}
    for i, (n_clusters, cols) in enumerate(zip(n_clusters_list, cols_list)):
        df_part = df.iloc[:, cols]
        if heat_on_column not in df_part.columns:
            raise ValueError(f"La colonne '{heat_on_column}' n'est pas dans les colonnes sélectionnées.")
        df_heat = df_part[df_part[heat_on_column] == 1].drop(columns=[heat_on_column])
        model = apply_kmeans(n_clusters=n_clusters, data=df_heat)
        df_part.loc[df_part[heat_on_column] == 1, 'clusters'] = model.labels_
        df_part.loc[df_part[heat_on_column] == 0, 'clusters'] = n_clusters
        clustered_dfs[f'df_part{i+1}'] = df_part
    return clustered_dfs

"""
Évalue les performances des modèles sur des données multi-labels avec une séparation train-test.
"""    
    
def evaluate_models_split_multi_label(X_train, X_test, y_train, y_test, models):
    results = {}
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)
    for name, model in models.items():
        print(f"\nÉvaluation de {name} avec train_test_split...")
        start_time = time.time()
        model.fit(X_train, y_train_bin)
        y_pred_bin = model.predict(X_test)
        execution_time = time.time() - start_time
        f1 = f1_score(y_test_bin, y_pred_bin, average='weighted')
        accuracy = accuracy_score(y_test_bin, y_pred_bin)
        zero_one = zero_one_loss(y_test_bin, y_pred_bin)
        hamming = hamming_loss(y_test_bin, y_pred_bin)
        results[name] = {
            "f1_score": f1,
            "accuracy": accuracy,
            "zero_one_loss": zero_one,
            "hamming_loss": hamming,
            "execution_time (s)": execution_time
        }
        print(f"{name} - Accuracy: {accuracy:.4f} - F1 Score: {f1:.4f} - 0/1 Loss: {zero_one:.4f} - Hamming Loss: {hamming:.4f} - Temps d'exécution: {execution_time:.4f} sec")
        print("###################################################################")
    return results


"""
Évalue les performances des modèles sur des données multi-labels avec une séparation cross_validation.
""" 

def evaluate_models_cv_multi_label(X, y, models, cv=4):
    results = {}
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)
    for name, model in models.items():
        print(f"\nÉvaluation de {name} avec Cross Validation ({cv}-folds)...")
        start_time = time.time()
        scoring = ["accuracy", "f1_weighted"]
        scores = {}
        for metric in scoring:
            score = cross_val_score(model, X, y_bin, cv=KFold(n_splits=cv, shuffle=True, random_state=42), scoring=metric)
            scores[metric] = np.mean(score)        
        zero_one_scores = []
        hamming_scores = [] 
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train_bin, y_test_bin = y_bin[train_index], y_bin[test_index]
            model.fit(X_train, y_train_bin)
            y_pred_bin = model.predict(X_test)
            zero_one_scores.append(zero_one_loss(y_test_bin, y_pred_bin))
            hamming_scores.append(hamming_loss(y_test_bin, y_pred_bin))   
        scores["zero_one_loss"] = np.mean(zero_one_scores)
        scores["hamming_loss"] = np.mean(hamming_scores)
        execution_time = time.time() - start_time
        results[name] = scores
        results[name]["execution_time (s)"] = execution_time
        print(f"{name} - Accuracy: {scores['accuracy']:.4f} - F1 Score: {scores['f1_weighted']:.4f} - 0/1 Loss: {scores['zero_one_loss']:.4f} - Hamming Loss: {scores['hamming_loss']:.4f} - Temps d'exécution: {execution_time:.4f} sec")
        print("###################################################################")
    
    return results



def compute_cluster_centers(df_part):
    cols_temporal = [col for col in df_part.columns if col not in ["heat_on", "clusters"]]
    centers = df_part.groupby("clusters")[cols_temporal].mean()
    return centers






########################################################### deep learning


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
   
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss during training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss, mae, mse = model.evaluate(X_val, y_val)
    rmse = np.sqrt(mse) 
    print(f"Validation Loss : {loss}")
    print(f"Validation MAE: {mae}")
    print(f"Validation MSE: {mse}")
    print(f"Validation RMSE: {rmse}")
    predictions = model.predict(X_test)    
    mae_test = mean_absolute_error(y_test, predictions)
    mse_test = mean_squared_error(y_test, predictions)
    rmse_test = np.sqrt(mse_test)
    print(f"Test MAE: {mae_test}")
    print(f"Test MSE: {mse_test}")
    print(f"Test RMSE: {rmse_test}")
    return history, loss, mae, mse,rmse,  mae_test, mse_test, rmse_test,predictions




def predict_and_inverse_transform(model, sample_input, scaler_temp, scaler_cons): 
    temperature_input = sample_input[-25:-1]
    conso_input = sample_input[-49:-25]
    temperature_input = temperature_input.reshape(1, 24) 
    temperature_pred = scaler_temp.inverse_transform(temperature_input)
    sample_input = sample_input.reshape(1, 1, 74)
    prediction = model.predict(sample_input)
    conso_input = conso_input.reshape(1, 24)
    conso_reel = scaler_cons.inverse_transform(conso_input)
    conso_pred = scaler_cons.inverse_transform(prediction)  
    print("Températures d'aujourd'hui :", temperature_pred.flatten())
    print("Consommation d'aujourd'hui reel :", conso_reel.flatten())
    print("Consommation d'aujourd'hui prédite  :", conso_pred)

from sklearn.utils import resample
def downsample_majority_class(df, target_column):
    counts = df[target_column].value_counts()
    majority_value = counts.idxmax()
    max_other = counts.drop(index=majority_value).max()
    df_majority = df[df[target_column] == majority_value]
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=max_other, random_state=42)
    df_others = df[df[target_column] != majority_value]
    balanced_df = pd.concat([df_majority_downsampled, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df



def plot_true_vs_predicted(indices, y_test, predictions, scaler_cons):
    true_total = []
    pred_total = []
    x_ticks = []

    for idx in indices:
        conso_reel = scaler_cons.inverse_transform(y_test[idx].reshape(1, 24))
        conso_pred = scaler_cons.inverse_transform(predictions[idx].reshape(1, 24))
        true_total.extend(conso_reel.flatten())
        pred_total.extend(conso_pred.flatten())
        x_ticks.append(len(true_total)) 
        
    plt.figure(figsize=(14, 6))
    plt.plot(true_total, label='True values', color='blue')
    plt.plot(pred_total, label='Predicted values', color='orange')
    plt.title('True vs Predicted values')
    plt.xlabel('Temps (heures concaténées)')
    plt.ylabel('Consommation')


    for xt in x_ticks[:-1]:
        plt.axvline(x=xt, color='black', linestyle='--', alpha=0.3)
    n_points = len(true_total)
    x_labels = [str(i) for i in range(24)] 
    x_labels_repeated = (x_labels * (n_points // 24)) + x_labels[:n_points % 24]
    plt.xticks(ticks=[i * 24 for i in range(len(indices))], labels=x_labels_repeated[:len(x_ticks)])
    plt.legend()
    plt.tight_layout()
    plt.show()

