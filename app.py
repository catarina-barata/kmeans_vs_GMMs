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
using **Sepal Length** and **Sepal Width** from the Iris dataset. Experiment with different GMM covariance structures 
to see how cluster shapes adapt to the data!
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
seed = st.sidebar.number_input("Random Seed:", min_value=0, max_value=999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Model Differences")
st.sidebar.markdown("- **K-Means**: Circular/spherical hard boundaries.")
st.sidebar.markdown("- **GMM (Spherical)**: Equal variance across dimensions.")
st.sidebar.markdown("- **GMM (Diagonal)**: Axis-aligned elliptical boundaries.")
st.sidebar.markdown("- **GMM (Full)**: Rotated ellipses capturing correlations.")

# -----------------------------------------------------------------------------
# Helper Function to Draw GMM Covariance Ellipses
# -----------------------------------------------------------------------------
def draw_ellipse(position, covariance, cov_type, ax, **kwargs):
    if cov_type == "full":
        # 2D Covariance Matrix
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    elif cov_type == "diagonal":
        # 1D Array with 2 elements
        width, height = 2 * np.sqrt(covariance)
        angle = 0
    else:  # spherical
        # Scalar or 1-element array
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
# Model Fitting & Visualization
# -----------------------------------------------------------------------------
# 1. Fit K-Means
km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
km_labels = km.fit_predict(X)

# 2. Fit GMM
gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=seed)
gmm_labels = gmm.fit_predict(X)

col1, col2 = st.columns(2)

# Meshgrid for Voronoi / Decision Contour plotting
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
    ax1.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        c='red', marker='x', s=150, linewidth=3, label='Centroids', zorder=10
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
    
    # Plot component means and ellipses
    ax2.scatter(
        gmm.means_[:, 0], gmm.means_[:, 1],
        c='red', marker='X', s=150, edgecolor='black', label='Component Means', zorder=10
    )
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
