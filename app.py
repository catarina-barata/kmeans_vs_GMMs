import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="K-Means vs GMM Demo",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Comparison: K-Means vs Gaussian Mixture Models (GMM)")
st.markdown("""
This demo compares **K-Means** (hard boundary clustering) against **GMM** (soft probabilistic clustering) 
using **Sepal Length** and **Sepal Width** from the Iris dataset.
""")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length and Sepal width
    feature_names = ["Sepal Length (cm)", "Sepal Width (cm)"]
    return X, feature_names

X, feature_names = load_data()

# -----------------------------------------------------------------------------
# Sidebar Settings
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

k = st.sidebar.slider("Number of Clusters/Components (K):", min_value=2, max_value=6, value=3)
cov_type = st.sidebar.selectbox("Covariance Type:", ["spherical", "diag", "full"])
seed_val = st.sidebar.number_input("Random Seed:", min_value=0, max_value=999, value=42, step=1)
seed = int(seed_val)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Visual Legend")
st.sidebar.markdown("⭐ **Yellow Star:** Initial Centroids / Initial Means")
st.sidebar.markdown("❌ **Red Cross:** Final Centroids / Final Means")

# -----------------------------------------------------------------------------
# Helper Function to Draw Covariance Ellipses
# -----------------------------------------------------------------------------
def draw_ellipse(position, covariance, cov_type, ax, **kwargs):
    if cov_type == "full":
        # 2D Covariance Matrix (2, 2)
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    elif cov_type == "diag":
        # 1D Array with 2 elements [var_x, var_y]
        width = 2 * np.sqrt(covariance[0])
        height = 2 * np.sqrt(covariance[1])
        angle = 0
    else:  # spherical
        # Scalar value or 1-element array
        val = np.atleast_1d(covariance)[0]
        width = height = 2 * np.sqrt(val)
        angle = 0

    # Draw 1-std and 2-std contour ellipses
    for scale in [1.0, 2.0]:
        ellipse = Ellipse(
            xy=position,
            width=scale * width,
            height=scale * height,
            angle=angle,
            **kwargs
        )
        ax.add_patch(ellipse)

# Helper to estimate empirical cluster covariances for K-Means based on type
def get_kmeans_covariances(X, labels, k, cov_type):
    covs = []
    for i in range(k):
        cluster_pts = X[labels == i]
        if len(cluster_pts) <= 1:
            # Fallback for single point or empty cluster
            if cov_type == "full":
                covs.append(np.eye(2) * 1e-3)
            elif cov_type == "diag":
                covs.append(np.ones(2) * 1e-3)
            else:
                covs.append(1e-3)
            continue
            
        full_cov = np.cov(cluster_pts, rowvar=False)
        
        if cov_type == "full":
            covs.append(full_cov)
        elif cov_type == "diag":
            covs.append(np.diag(full_cov))
        elif cov_type == "spherical":
            covs.append(np.mean(np.diag(full_cov)))
            
    return covs

# -----------------------------------------------------------------------------
# Model Fitting
# -----------------------------------------------------------------------------
# 1. K-Means
km_init = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=seed)
km_init.fit(X)
km_initial_centroids = km_init.cluster_centers_

km = KMeans(n_clusters=k, init=km_initial_centroids, n_init=1, random_state=seed)
km_labels = km.fit_predict(X)

# Calculate empirical covariances for K-Means clusters
km_covariances = get_kmeans_covariances(X, km_labels, k, cov_type)

# 2. Gaussian Mixture Model (GMM)
gmm_initial_means = km_initial_centroids

gmm = GaussianMixture(
    n_components=int(k),
    covariance_type=str(cov_type),
    random_state=seed
)
gmm.fit(X)
gmm_labels = gmm.predict(X)

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

# Grid for decision boundary plots
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# --- Plot 1: K-Means ---
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    
    Z_km = km.predict(grid_points).reshape(xx.shape)
    ax1.contourf(xx, yy, Z_km, alpha=0.2, cmap='Set2')
    
    ax1.scatter(X[:, 0], X[:, 1], c=km_labels, cmap='Set2', edgecolor='k', s=50, alpha=0.8)
    
    # Starting Points
    ax1.scatter(
        km_initial_centroids[:, 0], km_initial_centroids[:, 1],
        c='yellow', marker='*', s=250, edgecolor='black', linewidth=1.5,
        label='Initial Centroids', zorder=10
    )
    # Final Centroids
    ax1.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        c='red', marker='x', s=150, linewidth=3,
        label='Final Centroids', zorder=10
    )
    
    # Draw K-Means Empirical Covariance Ellipses
    for i in range(k):
        draw_ellipse(km.cluster_centers_[i], km_covariances[i], cov_type, ax=ax1, alpha=0.25, color='black')
    
    ax1.set_title(f"K-Means (K = {k})", fontweight='bold')
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.legend(loc='upper right')
    
    st.pyplot(fig1)
    st.metric("Final Inertia (WCSS)", f"{km.inertia_:.2f}")

# --- Plot 2: GMM ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    
    Z_gmm = gmm.predict(grid_points).reshape(xx.shape)
    ax2.contourf(xx, yy, Z_gmm, alpha=0.2, cmap='Set2')
    
    ax2.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='Set2', edgecolor='k', s=50, alpha=0.8)
    
    # Starting Means
    ax2.scatter(
        gmm_initial_means[:, 0], gmm_initial_means[:, 1],
        c='yellow', marker='*', s=250, edgecolor='black', linewidth=1.5,
        label='Initial Means', zorder=10
    )
    # Final Means
    ax2.scatter(
        gmm.means_[:, 0], gmm.means_[:, 1],
        c='red', marker='x', s=150, linewidth=3,
        label='Final Means', zorder=10
    )
    
    # Draw GMM Covariance Ellipses
    for i in range(gmm.n_components):
        draw_ellipse(gmm.means_[i], gmm.covariances_[i], cov_type, ax=ax2, alpha=0.25, color='black')
        
    ax2.set_title(f"GMM ({cov_type.capitalize()} Covariance, K = {k})", fontweight='bold')
    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1])
    ax2.legend(loc='upper right')
    
    st.pyplot(fig2)
    st.metric("Log-Likelihood", f"{gmm.score(X) * len(X):.2f}")
