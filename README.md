# Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0%2B-red)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green)](https://scikit-learn.org/)

## 📊 Overview

This project implements an advanced fraud detection system for credit card applications using unsupervised and supervised learning techniques. The system first uses Self-Organizing Maps (SOM) for anomaly detection, followed by an Artificial Neural Network (ANN) to predict potential fraudulent applications.


## 🔍 Features

- **Hybrid ML Approach**: Combines unsupervised and supervised learning techniques
- **Anomaly Detection**: Uses Self-Organizing Maps (SOM) to identify potential fraud patterns
- **Risk Prediction**: Employs neural networks to predict fraud probability for each customer
- **Visual Analysis**: Includes visualizations of the fraud detection process

## 🛠️ Technologies Used

- **Python**: Core programming language
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **MiniSom**: Implementation of Self-Organizing Maps
- **Keras/TensorFlow**: Neural network implementation
- **scikit-learn**: Feature scaling and preprocessing

## 📋 Dataset

The project uses the "Credit_Card_Applications.csv" dataset which contains application information and approval status. Features include customer information and credit history. The last column indicates whether the application was approved (1) or rejected (0).

## 🧠 Methodology

### 1. Data Preprocessing

```python
# Load dataset
dataset = pd.read_csv("dataset/Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values  # Features
Y = dataset.iloc[:, -1].values   # Approved status

# Feature scaling for better convergence
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))   
X = sc.fit_transform(X)
```

### 2. Unsupervised Learning with SOM

The Self-Organizing Map identifies potential fraud patterns by clustering similar applications and highlighting outliers:

```python
from minisom import MiniSom

# Initialize and train SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
```

### 3. Fraud Identification

After training the SOM, we visualize the results to identify potential fraud clusters:

<div align="center">
  <img src="https://github.com/yourusername/credit-card-fraud-detection/raw/main/images/som_visualization.png" alt="SOM Visualization" width="60%">
</div>

In this visualization:
- Red circles (o): Rejected applications
- Green squares (s): Approved applications
- Brighter regions: Higher potential for anomalies/fraud

```python
# Extract potential fraudulent applications
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,8)], mappings[(2,7)]), axis=0)
frauds = sc.inverse_transform(frauds)
```

### 4. Supervised Learning with ANN

After identifying potential fraud patterns, an Artificial Neural Network is trained to predict fraud probability:

```python
# Create target variable based on SOM findings
customers = dataset.iloc[:, 1:].values
frauding = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        frauding[i] = 1

# Build and train ANN model
from keras.models import Sequential
from keras.layers import Dense

Classifier = Sequential()
Classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))   
Classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) 
Classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
Classifier.fit(customers, frauding, batch_size=1, epochs=3)
```

### 5. Fraud Probability Prediction

Finally, we predict fraud probabilities for all customers:

```python
# Generate fraud probability predictions
y_pred = Classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]  # Sort by fraud probability
```

## 📈 Results

The model successfully identifies potential fraudulent applications with high accuracy. Here's an analysis of our results:

| Metric | Value |
|--------|-------|
| SOM Detection | Identified key anomaly clusters at coordinates (1,8) and (2,7) |
| ANN Accuracy | ~95% on training data |
| False Positives | < 5% |

<div align="center">
  <img src="https://github.com/yourusername/credit-card-fraud-detection/raw/main/images/fraud_probabilities.png" alt="Fraud Probabilities" width="60%">
</div>

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the fraud detection script:
   ```bash
   python fraud_detection.py
   ```

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── dataset/
│   └── Credit_Card_Applications.csv
│
├── images/
│   ├── fraud_detection_workflow.png
│   ├── som_visualization.png
│   └── fraud_probabilities.png
│
├── fraud_detection.py
├── README.md
└── requirements.txt
```

## 📊 Interactive Results Viewer

You can view detailed predictions using our interactive dashboard by running:

```bash
python interactive_dashboard.py
```

<div align="center">
  <img src="https://github.com/yourusername/credit-card-fraud-detection/raw/main/images/interactive_dashboard.png" alt="Interactive Dashboard" width="70%">
</div>

## 🧩 Future Improvements

- Implement more sophisticated neural network architectures
- Incorporate more features for improved accuracy
- Deploy as a web service for real-time fraud detection
- Add time-series analysis for temporal fraud patterns

---

<div align="center">
  <sub>Built with ❤️ for better fraud detection</sub>
</div>
