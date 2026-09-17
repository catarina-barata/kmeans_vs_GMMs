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
cov_type = st.sidebar.selectbox("GMM Covariance Type:", ["spherical", "diagonal", "full"])

# Ensure seed is strictly a Python int
seed_val = st.sidebar.number_input("Random Seed:", min_value=0, max_value=999, value=42, step=1)
seed = int(seed_val)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Visual Legend")
st.sidebar.markdown("⭐ **Yellow Star:** Initial Centroids / Initial Means")
st.sidebar.markdown("❌ **Red Cross:** Final Centroids / Final Means")

# -----------------------------------------------------------------------------
# Helper Function to Draw GMM Covariance Ellipses
# -----------------------------------------------------------------------------
def draw_ellipse(position, covariance, cov_type, ax, **kwargs):
    if cov_type == "full":
        # 2D Covariance Matrix (2, 2)
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    elif cov_type == "diagonal":
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

# -----------------------------------------------------------------------------
# Model Fitting
# -----------------------------------------------------------------------------
# 1. K-Means
km_init = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=seed)
km_init.fit(X)
km_initial_centroids = km_init.cluster_centers_

km = KMeans(n_clusters=k, init=km_initial_centroids, n_init=1, random_state=seed)
km_labels = km.fit_predict(X)

# 2. Gaussian Mixture Model (GMM)
# Extract starting means using standard KMeans initialization
gmm_initial_means = km_initial_centroids

# Instantiate GMM cleanly with validated string and int types
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
    
    # Draw Ellipses
    for i in range(gmm.n_components):
        draw_ellipse(gmm.means_[i], gmm.covariances_[i], cov_type, ax=ax2, alpha=0.25, color='black')
        
    ax2.set_title(f"GMM ({cov_type.capitalize()} Covariance, K = {k})", fontweight='bold')
    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1])
    ax2.legend(loc='upper right')
    
    st.pyplot(fig2)
    st.metric("Log-Likelihood", f"{gmm.score(X) * len(X):.2f}")

# -----------------------------------------------------------------------------
# Model Selection Criteria
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 GMM Model Quality Indicators")
col_m1, col_m2, col_m3 = st.columns(3)

col_m1.metric("AIC (Akaike Information Criterion)", f"{gmm.aic(X):.2f}")
col_m2.metric("BIC (Bayesian Information Criterion)", f"{gmm.bic(X):.2f}")
col_m3.metric("Convergence Status", "Converged" if gmm.converged_ else "Failed")
