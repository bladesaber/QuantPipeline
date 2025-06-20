"""
Theme of codependence distance metrics:
inner pattern correlation, difference between vector, mixed distance,
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, entropy
from sklearn import metrics
from typing import Callable


def get_optimal_number_of_bins(n_observations: int, corr_coef: float) -> int:
    """
    Get the optimal number of bins for a histogram.
    """
    return int((np.log2(n_observations) + 1) * (1 - 2 * np.abs(corr_coef)))


class DistanceMetric:
    @staticmethod
    def angular_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the angular distance between two vectors.
        """
        corr_coef = np.corrcoef(x, y)[0][1]
        return np.sqrt(0.5 * (1 - corr_coef))

    @staticmethod
    def absolute_angular_distance(x: np.ndarray, y: np.ndarray) -> float:
        corr_coef = np.corrcoef(x, y)[0][1]
        return np.sqrt(0.5 * (1 - abs(corr_coef)))

    @staticmethod
    def squared_angular_distance(x: np.ndarray, y: np.ndarray) -> float:
        corr_coef = np.corrcoef(x, y)[0][1]
        return np.sqrt(0.5 * (1 - corr_coef ** 2))

    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum((x - y) ** 2))

    @staticmethod
    def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(x - y))

    @staticmethod
    def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.max(np.abs(x - y))


class CorrelationMetric:
    """range: [-1, 1]"""
    
    @staticmethod
    def inner_pattern_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Returns distance correlation between two vectors. Distance correlation captures both linear and non-linear
        dependencies.

        Formula used for calculation:

        Distance_Corr[X, Y] = dCov[X, Y] / (dCov[X, X] * dCov[Y, Y])^(1/2)

        dCov[X, Y] is the average Hadamard product of the doubly-centered Euclidean distance matrices of X, Y.

        Read Cornell lecture notes for more information about distance correlation:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
        """
        # Ensure inputs are 2D arrays
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = x.shape[0]
        
        # Compute distance matrices directly using broadcasting
        # This is faster than using pdist and squareform
        a = np.abs(x - x.T)
        b = np.abs(y - y.T)
        
        print(a)
        print(b)
        
        # Compute means once
        a_mean = a.mean()
        b_mean = b.mean()
        a_row_mean = a.mean(axis=1)
        a_col_mean = a.mean(axis=0)
        b_row_mean = b.mean(axis=1)
        b_col_mean = b.mean(axis=0)
        
        # Center matrices in one operation
        A = a - a_row_mean[:, None] - a_col_mean[None, :] + a_mean
        B = b - b_row_mean[:, None] - b_col_mean[None, :] + b_mean
        
        # Compute covariances using einsum for better performance    
        d_cov_xx = np.einsum('ij,ij->', A, A) / (n * n)  # (A * A).sum() / (n * n)
        d_cov_xy = np.einsum('ij,ij->', A, B) / (n * n)  # (A * B).sum() / (n * n)
        d_cov_yy = np.einsum('ij,ij->', B, B) / (n * n)  # (B * B).sum() / (n * n)
        
        coef = np.sqrt(d_cov_xy) / np.sqrt(np.sqrt(d_cov_xx) * np.sqrt(d_cov_yy))
        return 1.0 - coef

    @staticmethod
    def spearmans_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates a statistical estimate of Spearman's rho, Pearson correlation applied to ranked data
        Formula for calculation:
        rho = 1 - (6)/(T*(T^2-1)) * Sum((X_t-Y_t)^2)
        1. convert x and y to ranks
        2. calculate the difference between the ranks
        3. calculate the sum of the squared differences
        4. calculate the Spearman's rho
        """
        rho, _ = spearmanr(x, y)
        return 0.5 * (1.0 - rho)  # Convert to [0,1] range where 0 means more correlated

    @staticmethod
    def z_perason_correlation(x: np.ndarray, y: np.ndarray, dim_mean: np.ndarray, dim_std: np.ndarray) -> float:
        x_z = (x - dim_mean) / dim_std
        y_z = (y - dim_mean) / dim_std
        return 0.5 * (1.0 - np.corrcoef(x_z, y_z)[0][1])

    @staticmethod
    def kl_divergence(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(x * np.log(x / y))

    @staticmethod
    def js_divergence(x: np.ndarray, y: np.ndarray) -> float:
        return 0.5 * (DistanceMetric.kl_divergence(x, y) + DistanceMetric.kl_divergence(y, x))

    @staticmethod
    def histogram_correlation(x: np.ndarray, y: np.ndarray, bandwidth: float = 0.01) -> float:
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        bins = np.arange(min_val, max_val + bandwidth, bandwidth)
        x_hist = np.histogram(x, bins=bins, density=True)[0]
        y_hist = np.histogram(y, bins=bins, density=True)[0]
        return DistanceMetric.js_divergence(x_hist, y_hist)

    @staticmethod
    def cosine_correlation(x: np.ndarray, y: np.ndarray) -> float:
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    @staticmethod
    def mutual_information_correlation(x: np.ndarray, y: np.ndarray, n_bins: int = None, normalize: bool = True) -> float:
        """
        Returns mutual information (I) between two vectors.

        This function uses the discretization with the optimal bins algorithm proposed in the works of
        Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

        Read Cornell lecture notes for more information about the mutual information:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

        :param x: (np.array) X vector.
        :param y: (np.array) Y vector.
        :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                            (None by default)
        :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
        :return: (float) Mutual information score.
        """
        if n_bins is None:
            n_bins = DistanceMetric.get_optimal_number_of_bins(len(x), np.corrcoef(x, y)[0][1])
        contingency = np.histogram2d(x, y, bins=n_bins, density=False)[0]
        mutual_info = metrics.mutual_info_score(None, None, contingency=contingency)
        if normalize:
            entropy_x = entropy(np.histogram(x, bins=n_bins, density=True)[0])
            entropy_y = entropy(np.histogram(y, bins=n_bins, density=True)[0])
            max_mutual_info = min(entropy_x, entropy_y)
            mutual_info = mutual_info / max_mutual_info
        return 1 - mutual_info

    @staticmethod
    def variation_of_information_correlation(x: np.ndarray, y: np.ndarray, n_bins: int = None, normalize: bool = True) -> float:
        """
        Returns variantion of information (VI) between two vectors.

        This function uses the discretization using optimal bins algorithm proposed in the works of
        Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

        Read Cornell lecture notes for more information about the variation of information:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

        :param x: (np.array) X vector.
        :param y: (np.array) Y vector.
        :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                            (None by default)
        :param normalize: (bool) True to normalize the result to [0, 1]. (False by default)
        :return: (float) Variation of information score.
        """
        if n_bins is None:
            n_bins = DistanceMetric.get_optimal_number_of_bins(len(x), np.corrcoef(x, y)[0][1])
        contingency = np.histogram2d(x, y, bins=n_bins)[0]
        mutual_info = metrics.mutual_info_score(None, None, contingency=contingency)  # Mutual information
        marginal_x = entropy(np.histogram(x, bins=n_bins)[0])  # Marginal for x
        marginal_y = entropy(np.histogram(y, bins=n_bins)[0])  # Marginal for y
        score = marginal_x + marginal_y - 2 * mutual_info
        if normalize:
            joint_dist = marginal_x + marginal_y - mutual_info  # Joint distribution
            score /= joint_dist
        return score


def mixed_codependence_distance(dist0: float, dist1: float, theta: float) -> float:
    return (1 - theta) * dist0 + theta * dist1


def compute_codependence_matrix(df: pd.DataFrame, distance_func: Callable) -> pd.DataFrame:
    features_cols = df.columns.values
    n = df.shape[1]
    np_df = df.values.T
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = distance_func(np_df[i], np_df[j])
                    
    # Make matrix symmetrical
    mat = mat + mat.T
  
    #  Dependence_matrix converted into a DataFrame.
    cod_df = pd.DataFrame(data=mat, index=features_cols, columns=features_cols)
    return cod_df

