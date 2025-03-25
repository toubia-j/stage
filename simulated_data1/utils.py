import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score



def apply_kmeans(n_clusters,data):
    
    kmeans=KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans 


def draw(data, text, X_label,Y_label):
    
    plt.figure(figsize=(10, 6))
    for index, row in data.iterrows():
        plt.plot(range(0, 24), row, label=f"Jour {index}")
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(text)
    plt.tight_layout()
    plt.show()


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



def evaluate_clustering_cooling(consommation_fr):
   
    labels = consommation_fr['clusters']
    consommation3_fr = consommation_fr.drop(columns=["clusters", "cool_on"])
    sil_score = silhouette_score(consommation3_fr, labels, metric='euclidean')
    db_score = davies_bouldin_score(consommation3_fr, labels)
    print(f"Davies-Bouldin Index: {db_score}")
    print(f"Silhouette Score: {sil_score}")
