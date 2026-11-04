# High-Dimensional Classification: Custom k-NN vs. Decision Trees

This project evaluates distance-based vs. rule-based classification on high-dimensional pixel data for **CM3015: Machine Learning and Neural Networks**[cite: 24, 25]. 

To explore the mechanics behind distance metrics under the curse of dimensionality, I implemented a $k$-Nearest Neighbors ($k$-NN) classifier entirely from scratch using pure NumPy and evaluated it against scikit-learn's optimized Decision Tree classifier on the 64-dimensional Handwritten Digits dataset ($8 \times 8$ grayscale images)[cite: 24, 25].

---

## Core Experiments & Methodology

* **Custom Vectorized $k$-NN:** Built an estimator from scratch inheriting from `BaseEstimator` and `ClassifierMixin`[cite: 27]. It calculates high-dimensional Euclidean distances, extracts the top-$k$ nearest neighbors via `np.argsort`, and determines the final class label using a majority vote[cite: 25, 27].
* **PCA Dimensionality Reduction:** Visualized the 64-dimensional feature space by projecting it down to 2 principal components[cite: 25, 26]. This highlights which digits form tight clusters and which classes overlap significantly in pixel space[cite: 26].
* **Feature Scaling:** Used `StandardScaler` to normalize pixel intensity variance across features, ensuring uniform Euclidean distance weighting across all 64 dimensions[cite: 26].
* **Hyperparameter Tuning & Cross-Validation:** Systematically swept odd values of $k$ from 1 to 15 to assess overfitting versus over-smoothing[cite: 26]. Used 5-fold cross-validation on the Decision Tree to get an accurate benchmark of its performance[cite: 27].

---

## Results & Error Analysis

| Classifier | Hyperparameters | Test Accuracy | Characteristics |
| :--- | :--- | :--- | :--- |
| **Custom NumPy $k$-NN** | $k=1$, Euclidean Distance[cite: 25, 28] | **97.78%**[cite: 28] | Maps continuous pixel transitions accurately; high test-time compute[cite: 28] |
| **Decision Tree (CART)** | Max Depth = 10, Gini Impurity[cite: 25, 28] | 85.00%[cite: 28] | Fast inference; struggles with axis-aligned splits on smooth curves[cite: 25, 28] |

### Why $k$-NN Outperforms the Tree on Pixel Data
Handwritten digits occupy a continuous manifold in pixel space—an image of a '3' shares close Euclidean distance with other variations of '3'[cite: 28]. Decision trees attempt to partition this space using orthogonal, axis-aligned cuts, which struggle to approximate smooth circular digit strokes without overfitting to specific training set quirks[cite: 25, 28].

---

## Code Structure

* **`MachineLearning_Mids.ipynb`**: Complete notebook containing the from-scratch $k$-NN implementation, PCA projections, hyperparameter sweeps, and confusion matrix evaluations[cite: 26, 27, 28, 30].
* **`requirements.txt`**: Minimal Python dependencies (NumPy, SciPy, scikit-learn, Matplotlib, Seaborn)[cite: 24, 29].

---

## Quickstart

Clone the repository:
```bash
git clone [https://github.com/Junaid-010/high-dimensional-ml-benchmarks.git](https://github.com/Junaid-010/high-dimensional-ml-benchmarks.git)
cd high-dimensional-ml-benchmarks