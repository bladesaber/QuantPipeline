"""
Optimal Number of Clusters (ONC Algorithm)
Detection of False Investment Strategies using Unsupervised Learning Methods
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
from sklearn.metrics import silhouette_samples


class OncCluster:
    @staticmethod
    def _get_random_seed():
        # get random state from system time
        return int(time.time())
    
    @staticmethod
    def _union_clusters(corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict):
        # Pre-allocate arrays for better performance
        n_clusters = len(clusters) + len(top_clusters)
        clusters_new = {}
        new_idx = []
        
        # Combine clusters more efficiently
        for i, cluster in enumerate(clusters.values()):
            clusters_new[i] = list(cluster)
            new_idx.extend(cluster)
            
        for i, cluster in enumerate(top_clusters.values(), start=len(clusters)):
            clusters_new[i] = list(cluster)
            new_idx.extend(cluster)
        
        corr_new = corr_mat.loc[new_idx, new_idx]
        
        corr_filled = corr_new.values
        np.fill_diagonal(corr_filled, 1)
        dist = np.sqrt((1 - corr_filled) / 2.0)
        
        kmeans_labels = np.zeros(len(dist), dtype=np.int32)
        for i, cluster in clusters_new.items():
            idxs = np.array([corr_new.index.get_loc(k) for k in cluster])
            kmeans_labels[idxs] = i
        
        silh_scores_new = pd.Series(
            silhouette_samples(dist, kmeans_labels),
            index=corr_new.index
        )
        
        return corr_new, clusters_new, silh_scores_new
        
        
    @staticmethod
    def _cluster_kmeans_base(corr_mat: pd.DataFrame, max_num_clusters: int = 10, repeat: int = 10):
        np.fill_diagonal(corr_mat.values, 1)
        dist_mat = ((1 - corr_mat) / 2.0) ** 0.5
        silh = pd.Series(dtype=np.float64)
        
        for _ in range(repeat):
            for num_clusters in range(2, max_num_clusters + 1):
                kmeans = KMeans(n_clusters=num_clusters, random_state=OncCluster._get_random_seed(), n_init=1)
                kmeans.fit(dist_mat)
                silh_ = silhouette_samples(dist_mat, kmeans.labels_)
                
                score = silh_.mean() / silh_.std()
                if silh.empty or score > silh.mean() / silh.std():
                    silh = silh_
                    kmeans_ = kmeans
        
        # Number of clusters equals to length(kmeans labels)
        new_idx = np.argsort(kmeans.labels_)

        # Reorder rows
        corr1 = corr_mat.iloc[new_idx]
        # Reorder columns
        corr1 = corr1.iloc[:, new_idx]
        
        # Cluster members
        clusters = {}
        for i in np.unique(kmeans.labels_):
            clusters[i] = corr_mat.columns[np.where(kmeans.labels_ == i)[0]].tolist()
        
        silh = pd.Series(silh, index=dist_mat.index)
        return corr1, clusters, silh
    
    def fit(self, corr_mat: pd.DataFrame, max_num_clusters: int = 10, repeat: int = 10):
        assert not np.any(corr_mat.isna()), "corr_mat contains NaN"
        assert not np.any(corr_mat.isin([np.inf, -np.inf])), "corr_mat contains infinite"
        
        max_non_collinear_col_num = corr_mat.drop_duplicates().shape[0]
        max_non_collinear_row_num = corr_mat.drop_duplicates().shape[1]
        max_num_clusters = min(max_non_collinear_col_num, max_non_collinear_row_num) - 1
        
        corr1, clusters, silh = OncCluster._cluster_kmeans_base(
            corr_mat, max_num_clusters=max_num_clusters, repeat=repeat
        )
        
        cluster_quality = {}
        for i in clusters.keys():
            if np.std(silh[clusters[i]]) == 0:
                cluster_quality[i] = float('Inf')
            else:
                cluster_quality[i] = np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]])
        avg_quality = np.mean(cluster_quality.values())
        
        redo_clusters = [i for i in cluster_quality.keys() if cluster_quality[i] < avg_quality]
        if len(redo_clusters) <= 2:
            # If 2 or less clusters have a quality rating less than the average then stop.
            return corr1, clusters, silh
        else:
            keys_redo = []
            for i in redo_clusters:
                keys_redo.extend(clusters[i])
                
            corr_tmp = corr_mat.loc[keys_redo, keys_redo]
            mean_redo_tstat = np.mean([cluster_quality[i] for i in redo_clusters])
            
            _, top_clusters, _ = OncCluster.cluster_kmeans_top(corr_tmp, repeat=repeat)
            corr_new, clusters_new, silh_new = OncCluster._union_clusters(
                corr_mat,
                {i: clusters[i] for i in clusters.keys() if i not in redo_clusters},
                top_clusters
            )
            
            new_tstat_mean = np.mean([
                np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) 
                for i in clusters_new
            ])
            
            if new_tstat_mean > mean_redo_tstat:
                return corr1, clusters_new, silh_new
            else:
                return corr1, clusters, silh

