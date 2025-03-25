import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold ,cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time

def extract_columns(filepath, column_index=4):
    df = pd.read_csv(filepath, delimiter="\t")
    values = df.iloc[:, column_index].values
    return pd.DataFrame(values.reshape(-1, 24))



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


def plot_clusters(consommation):  
    min_val = consommation.iloc[:, :-1].min().min() 
    max_val = consommation.iloc[:, :-1].max().max()
    ylim = [min_val - 2, max_val + 2]      
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4): 
        ax = axes[i // 2, i % 2] 
        cluster_data = consommation[consommation["clusters"] == i]     
        for index, row in cluster_data.iterrows():
            ax.plot(range(24), row.iloc[:-1], color='gray', alpha=0.5)            
        center = cluster_data.iloc[:, :-1].mean(axis=0) 
        ax.plot(range(24), center, color='red', label=f'Cluster {i} ({len(cluster_data)})')
        ax.set_xlim([0, 24])  
        ax.set_ylim(ylim) 
        ax.set_title(f"Cluster {i}")
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



