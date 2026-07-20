"""
IDEST: Intrinsic Dimension Estimation for Self-Supervised Learning

This implementation follows the paper:
"IDEST: Assessing Self-Supervised Learning Representations via Intrinsic Dimension"
Mordacq et al., 2026

The key insight: Lower intrinsic dimension (ID) indicates better SSL representations.
ID is estimated via the scaling behavior of Minimum Spanning Tree (MST) lengths.

Mathematical foundation:
    L(MST(X_n)) ∝ n^(d-1)/d
    where:
        - L(MST(X_n)) = total edge length of MST on n samples
        - d = intrinsic dimension
        - n = number of samples
    
    Taking logs:
        log(L) = ((d-1)/d) * log(n) + C
        slope m = (d-1)/d
        therefore: d = 1 / (1 - m)
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class IDEST:
    """
    IDEST: Intrinsic Dimension Estimation for SSL using Minimum Spanning Trees
    
    This class implements the dim_MST estimator and applies it to SSL representations.
    Lower estimated dimension = better representation quality.
    
    Parameters
    ----------
    n_subsample : int, optional (default=None)
        Maximum number of samples to use. If None, uses all samples.
        Paper uses 50,000 for ImageNet. For smaller datasets, use all.
    
    n_min : int, optional (default=10)
        Minimum sample size for log-log regression.
        Should be small enough to get multiple data points.
    
    step : int, optional (default=10)
        Step size between sample sizes.
        Smaller = more regression points = more stable estimate.
    
    n_runs : int, optional (default=5)
        Number of runs for stable estimation (due to random sampling).
        Higher = more stable but slower.
    
    random_state : int, optional (default=42)
        Seed for reproducibility.
    """
    
    def __init__(self, n_subsample=None, n_min=10, step=10, n_runs=5, random_state=42):
        self.n_subsample = n_subsample
        self.n_min = n_min
        self.step = step
        self.n_runs = n_runs
        self.random_state = random_state
        np.random.seed(random_state)
    
    def compute_mst_length(self, features):
        """
        Compute the total length of the Minimum Spanning Tree (MST) for a set of points.
        
        Parameters
        ----------
        features : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors (e.g., SSL representations)
        
        Returns
        -------
        mst_length : float
            Total length of the MST (sum of all edge weights)
        
        Notes
        -----
        The MST is the tree that connects all points with minimum total edge length.
        For points on a d-dimensional manifold, the MST length scales as:
            L(MST) ∝ n^(d-1)/d
        """
        # Step 1: Compute pairwise Euclidean distances between all points
        # This creates an n x n matrix where dist[i,j] = ||x_i - x_j||_2
        dist_matrix = distance_matrix(features, features)
        
        # Step 2: Compute the Minimum Spanning Tree
        # minimum_spanning_tree returns a sparse matrix representation
        # The MST connects all points with exactly n-1 edges
        mst_matrix = minimum_spanning_tree(dist_matrix)
        
        # Step 3: Sum all edge weights to get total length
        # This is L(MST(X)) in the paper
        mst_length = mst_matrix.sum()
        
        return mst_length
    
    def estimate_dimension_single_run(self, features):
        """
        Perform a single IDEST estimation run (stochastic due to subsampling).
        
        Parameters
        ----------
        features : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors to estimate dimension for
        
        Returns
        -------
        d : float
            Estimated intrinsic dimension
        slope : float
            Slope from log-log regression (used for debugging)
        r2 : float
            R-squared of the regression fit (quality indicator)
        sample_sizes : list
            Sample sizes used for regression
        mst_lengths : list
            Corresponding MST lengths
        """
        n_total = features.shape[0]
        
        # Step 1: Subsample if dataset is too large
        # The paper uses N=50,000 for computational tractability
        if self.n_subsample is not None and n_total > self.n_subsample:
            indices = np.random.choice(n_total, self.n_subsample, replace=False)
            features = features[indices]
            n_total = self.n_subsample
        
        # Step 2: Generate increasing sample sizes for log-log regression
        # We need enough points for regression (at least 5)
        sample_sizes = list(range(self.n_min, n_total, self.step))
        if sample_sizes[-1] != n_total:
            sample_sizes.append(n_total)
        
        # Ensure we have at least 5 sample sizes for regression
        if len(sample_sizes) < 5:
            # Create more fine-grained sampling
            sample_sizes = np.linspace(
                self.n_min, n_total, 
                min(10, n_total - 1), 
                dtype=int
            ).tolist()
            if n_total not in sample_sizes:
                sample_sizes.append(n_total)
            sample_sizes = sorted(set(sample_sizes))
        
        mst_lengths = []
        
        # Step 3: Compute MST length for each sample size
        # This is the core computation - O(n^2) for each sample size
        for n in sample_sizes:
            # Randomly sample n points without replacement
            indices = np.random.choice(n_total, n, replace=False)
            sample = features[indices]
            
            # Compute MST length for this subsample
            length = self.compute_mst_length(sample)
            mst_lengths.append(length)
        
        # Step 4: Log-log linear regression
        # We expect: log(L) = m * log(n) + b
        # where m = (d-1)/d
        log_n = np.log(sample_sizes).reshape(-1, 1)
        log_length = np.log(mst_lengths).reshape(-1, 1)
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(log_n, log_length)
        slope = reg.coef_[0][0]  # m in the paper
        intercept = reg.intercept_[0]  # log(C) in the paper
        r2 = reg.score(log_n, log_length)
        
        # Step 5: Extract intrinsic dimension from slope
        # d = 1 / (1 - m)
        # This comes from: m = (d-1)/d
        d = 1.0 / (1.0 - slope)
        
        return d, slope, r2, sample_sizes, mst_lengths
    
    def estimate_dimension(self, features, verbose=True):
        """
        Estimate intrinsic dimension with multiple runs for stability.
        
        Parameters
        ----------
        features : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors to estimate dimension for
        verbose : bool, optional (default=True)
            Print progress and results
        
        Returns
        -------
        d : float
            Estimated intrinsic dimension (average over runs)
        d_std : float
            Standard deviation across runs
        stats : dict
            Additional statistics (slope, R^2, etc.)
        """
        # Optional: Normalize features for better distance computation
        # This can improve stability but may affect dimension estimates
        # features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        d_values = []
        slopes = []
        r2_values = []
        all_sample_sizes = []
        all_mst_lengths = []
        
        if verbose:
            print(f"Estimating IDEST for {features.shape[0]} samples with {features.shape[1]} dimensions")
            print(f"Running {self.n_runs} stochastic runs for stability...")
        
        for run in range(self.n_runs):
            # Different random seed for each run
            np.random.seed(self.random_state + run * 100)
            
            d, slope, r2, sample_sizes, mst_lengths = self.estimate_dimension_single_run(features)
            
            d_values.append(d)
            slopes.append(slope)
            r2_values.append(r2)
            
            if verbose:
                print(f"  Run {run+1}: d={d:.2f}, slope={slope:.4f}, R²={r2:.4f}")
        
        # Compute statistics
        d_mean = np.mean(d_values)
        d_std = np.std(d_values)
        slope_mean = np.mean(slopes)
        r2_mean = np.mean(r2_values)
        
        if verbose:
            print(f"\nIDEST Results:")
            print(f"  Intrinsic Dimension: {d_mean:.2f} ± {d_std:.2f}")
            print(f"  Slope: {slope_mean:.4f}")
            print(f"  R²: {r2_mean:.4f}")
            print(f"  Note: Lower dimension = better representation quality")
        
        # Store results for later use (e.g., plotting)
        self.last_results = {
            'd': d_mean,
            'd_std': d_std,
            'slope': slope_mean,
            'r2': r2_mean,
            'd_values': d_values,
            'sample_sizes': all_sample_sizes if all_sample_sizes else sample_sizes,
            'mst_lengths': all_mst_lengths if all_mst_lengths else mst_lengths,
        }
        
        return d_mean, d_std, self.last_results
    
    def compare_models(self, model_features_dict, verbose=True):
        """
        Compare multiple SSL models by their IDEST scores.
        
        Parameters
        ----------
        model_features_dict : dict
            Dictionary mapping model names to feature matrices
        verbose : bool, optional (default=True)
            Print comparison results
        
        Returns
        -------
        results : dict
            Dictionary with IDEST scores for each model
        best_model : str
            Name of model with lowest IDEST (best representation)
        """
        results = {}
        
        if verbose:
            print("=" * 60)
            print("IDEST Model Comparison")
            print("=" * 60)
        
        for name, features in model_features_dict.items():
            d, d_std, _ = self.estimate_dimension(features, verbose=False)
            results[name] = {'mean': d, 'std': d_std}
            if verbose:
                print(f"{name:20s}: IDEST = {d:.2f} ± {d_std:.2f}")
        
        # Find best model (lowest dimension)
        best_model = min(results, key=lambda x: results[x]['mean'])
        
        if verbose:
            print("-" * 60)
            print(f"Best model: {best_model} (lowest intrinsic dimension)")
            print(f"IDEST score: {results[best_model]['mean']:.2f}")
            print("=" * 60)
        
        return results, best_model


def plot_idest_regression(features, title="IDEST Log-Log Regression", save_path=None):
    """
    Visualize the log-log regression used by IDEST.
    
    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Quick IDEST estimation for plotting
    idest = IDEST(n_min=5, step=5, n_runs=1)
    _, _, _, sample_sizes, mst_lengths = idest.estimate_dimension_single_run(features)
    
    # Log transform
    log_n = np.log(sample_sizes)
    log_L = np.log(mst_lengths)
    
    # Fit regression
    reg = LinearRegression()
    reg.fit(log_n.reshape(-1, 1), log_L.reshape(-1, 1))
    slope = reg.coef_[0][0]
    r2 = reg.score(log_n.reshape(-1, 1), log_L.reshape(-1, 1))
    d = 1.0 / (1.0 - slope)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Linear scale
    axes[0].plot(sample_sizes, mst_lengths, 'bo-', alpha=0.7)
    axes[0].set_xlabel('Sample Size (n)')
    axes[0].set_ylabel('MST Length L(MST)')
    axes[0].set_title('MST Length vs Sample Size')
    axes[0].grid(True, alpha=0.3)
    
    # Right: Log-log scale with regression line
    axes[1].scatter(log_n, log_L, color='blue', s=50, label='Data points')
    axes[1].plot(log_n, reg.predict(log_n.reshape(-1, 1)), 'r-', 
                 linewidth=2, label=f'Fit: slope={slope:.3f}')
    axes[1].set_xlabel('log(Sample Size)')
    axes[1].set_ylabel('log(MST Length)')
    axes[1].set_title(f'Log-Log Regression: d = {d:.2f}, R² = {r2:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return d, r2
