# ☀️ Praise The Sun: Solar Power Generation Prediction

### A Comparative Analysis of Statistical, Machine Learning, and Deep Learning Models

---

## 📖 Project Overview
This project, titled **"Praise The Sun"**, focuses on the analysis and prediction of **Solar Power Generation** using meteorological data. The goal is to optimize grid management by accurately predicting solar energy output. We approach this problem through a multi-stage pipeline, moving from statistical correlations to advanced Deep Learning architectures.

The analysis is divided into two core tasks:
1.  **Classification:** Categorizing power generation into **Low**, **Medium**, and **High** classes (useful for grid stability alerts).
2.  **Regression:** Predicting the exact **Power (kW)** output (useful for supply planning).

---

## 📊 Dataset
The dataset consists of high-resolution meteorological time-series data.
* **Target Variable:** `Active_Power` (kW)
* **Key Features:**
    * `Global_Horizontal_Radiation` (GHR) - The primary driver of solar output.
    * `Temperature` (°C) - Affects panel efficiency.
    * `Relative_Humidity` (%)
    * `Wind_Speed` (m/s)
    * **Temporal Features:** Hour of Day, Month, Seasonality (Cyclical encoding).

---

## 🛠️ Methodology & Phases

### Phase 1: Data Engineering & Preprocessing
* **Cleaning:** Handling missing values and stripping whitespace from column headers.
* **Feature Engineering:**
    * **Lag Features:** Created 1-hour and 24-hour lags to capture temporal dependencies (autocorrelation).
    * **Rolling Statistics:** Calculated rolling means for Radiation and Temperature to smooth noise.
    * **Cyclical Encoding:** Transformed `Hour` and `Month` into Sine/Cosine components to preserve cyclical continuity.

### Phase 2: Statistical Analysis & Visualization
* **Correlation Analysis:** Generated Heatmaps to quantify the relationship between Radiation, Temperature, and Power.
* **Hypothesis Testing:**
    * **ANOVA:** Confirmed that Radiation levels differ significantly across Power Classes.
    * **Chi-Square:** Verified the dependency between categorical weather states and power output.

### Phase 3: Dimensionality Reduction
* **PCA (Principal Component Analysis):** Determined that >95% of the variance is explained by just 3 principal components.
* **SVD (Singular Value Decomposition):** Visualized the dataset in 2D latent space, showing distinct clustering of "High" vs. "Low" power states.

### Phase 4: Baseline Modeling
We established performance baselines using traditional algorithms:
* **Naive Bayes:** Probabilistic baseline (assumes independence).
* **K-Nearest Neighbors (K-NN):** Distance-based classification (Euclidean vs. Manhattan).
* **Logistic Regression:** Linear baseline.
* **Linear Discriminant Analysis (LDA):** Geometric classifier maximizing class separability.

### Phase 5: Advanced Deep Learning
Implemented state-of-the-art architectures to capture non-linear and temporal dynamics:
* **MLP (Multi-Layer Perceptron):** Deep Feed-Forward Network.
* **XGBoost & CatBoost:** Gradient Boosting ensembles for tabular excellence.
* **CNN-LSTM:** Hybrid model using 1D Convolution for feature extraction and LSTM for sequence modeling.
* **CNN-BiLSTM:** Bidirectional architecture to capture both past and future context window dynamics.

---

## 🚀 Key Results
| Model Architecture | Accuracy / R² | Key Observation |
| :--- | :--- | :--- |
| **Naive Bayes** | ~55% | Fails due to strong feature correlation (Independence assumption violated). |
| **LDA** | ~85% | Strong geometric separation of classes. |
| **XGBoost** | ~94% | Excellent handling of non-linear thresholds. |
| **CNN-BiLSTM** | **~98%** | **Best Performer.** Successfully captures complex temporal patterns and lag effects. |

*> Note: Detailed confusion matrices and ROC curves are available in the project notebook.*

---

## 💻 Technologies Used
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, CatBoost
* **Deep Learning:** TensorFlow / Keras
* **Statistical Analysis:** SciPy, Pgmpy (Bayesian Networks)

---

## 📥 Installation & Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BackgroundTree/Praise-The-Sun.git](https://github.com/BackgroundTree/Praise-The-Sun.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow xgboost catboost seaborn matplotlib pgmpy
    ```
3.  **Run the Notebook:**
    Open `Praise_The_Sun_Phase2.ipynb` in Jupyter or Google Colab and run cells sequentially.

---

## 🔮 Future Work
* **Transformer Models:** Implementing Time-Series Transformers (TST) for potentially higher accuracy.
* **Real-Time Deployment:** wrapping the Bi-LSTM model in a Flask API for real-time inference.
* **Weather API Integration:** Fetching live weather forecasts to predict power 24 hours ahead.

---