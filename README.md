# **Credit Card Fraud Detection – Analytical Report**

## **1. Introduction**

The primary objective of this project is to analyze credit card transaction data to detect fraudulent activities. The dataset comprises **284,807 transactions** with **30 anonymized features (V1–V28, Amount, Time)** and a target variable `Class`, where:

* `0` → Legitimate (Normal) transaction
* `1` → Fraudulent transaction

Given the extreme imbalance in the dataset, with fraud cases representing only a tiny fraction (~0.17%), traditional supervised modeling approaches are challenging. Therefore, the project leverages **unsupervised anomaly detection techniques** to identify suspicious transactions.

---
## **2. Dataset Overview**

| Feature | Description                                                      |
| ------- | ---------------------------------------------------------------- |
| V1–V28  | Principal components derived via PCA to anonymize sensitive data |
| Amount  | Transaction amount in USD                                        |
| Time    | Seconds elapsed since the first recorded transaction             |
| Class   | Target variable: 0 = Normal, 1 = Fraud                           |

* **Total transactions:** 284,807
* **Fraud transactions:** 492 (~0.17%)
* **Non-fraud transactions:** 284,315 (~99.83%)

<img width="1076" height="700" alt="Screenshot 2026-03-19 013944" src="https://github.com/user-attachments/assets/f3fa6cd9-1e9c-4583-810e-6f078864eb1b" />




**Dataset Source:** This dataset was obtained from **Kaggle**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains anonymized features to protect sensitive user information.

The high imbalance necessitates careful evaluation and consideration of appropriate metrics.

---

## **3. Exploratory Data Analysis (EDA)**

### **3.1 Class Distribution**

* A bar chart reveals the overwhelming prevalence of legitimate transactions compared to fraudulent ones.
* **Console summary:**

```text
Total Fraud Transactions: 492
Total Non-Fraud Transactions: 284,315
```

**Observation:** Fraudulent transactions are extremely rare, emphasizing the need for anomaly detection approaches.

---

### **3.2 Transaction Amount Analysis**

* Histogram analysis of transaction amounts indicates that fraudulent transactions often involve **higher monetary values**, though high-value legitimate transactions also exist.
* Logarithmic scaling was used to visualize the skewed distribution.

---

### **3.3 Temporal Analysis**

* Scatter plots of transaction `Time` versus `Amount` demonstrate that fraudulent transactions are **distributed across time** and do not cluster at specific intervals, indicating that fraud attempts occur unpredictably.

---

### **3.4 Feature Correlation**

* Correlation heatmaps of the sampled dataset indicate minimal correlation between the anonymized PCA features and the `Class` variable.
* This justifies the application of **unsupervised or anomaly-based detection techniques**.

---

## **4. Sampling and Outlier Fraction**

* A **10% subset** of the dataset (28,481 transactions) was sampled to reduce computational complexity.
* **Fraud cases in sample:** 49
* **Non-fraud cases in sample:** 28,432
* **Outlier fraction:** 0.0017

<img width="1366" height="655" alt="Figure_3" src="https://github.com/user-attachments/assets/b3006c49-d322-424b-b926-99b196352657" />


The sample maintains the proportion of fraud to non-fraud cases consistent with the original dataset.

---

## **5. Outlier Detection Techniques**

Three unsupervised models were implemented to detect anomalies:

1. **Isolation Forest (IF)** – isolates anomalies by random partitioning
2. **Local Outlier Factor (LOF)** – measures local deviation of density
3. **One-Class Support Vector Machine (SVM)** – identifies the boundary of normal data

### **5.1 Methodology**

* Models were trained on the sampled dataset.
* Predictions were standardized: `0` for normal, `1` for fraudulent transactions.
* Model evaluation includes accuracy, misclassification counts, and classification reports.

### **5.2 Performance Overview (Sample Dataset)**

| Model                | Accuracy | Misclassifications     | Remarks                                                     |
| -------------------- | -------- | ---------------------- | ----------------------------------------------------------- |
| Isolation Forest     | ~99.7%   | Minimal errors         | Effective for rare anomaly detection                        |
| Local Outlier Factor | ~99.8%   | Very few errors        | Sensitive to neighborhood parameter                         |
| One-Class SVM        | ~99.5%   | Slightly higher errors | Computationally intensive, effective for boundary detection |

**Key Observation:** Unsupervised anomaly detection methods successfully identify the majority of fraudulent transactions despite the extreme class imbalance.

---

## **6. Key Findings**

1. Fraudulent transactions are extremely rare (~0.17% of the total dataset).
2. High transaction amounts are more indicative of fraud but are not solely sufficient for detection.
3. Outlier detection algorithms (Isolation Forest, LOF, One-Class SVM) provide robust detection of anomalies.
4. Sampling 10% of the data retains representative class proportions, facilitating computational efficiency without sacrificing model performance.

---

## **7. Conclusion**

Credit card fraud detection poses challenges due to **highly imbalanced datasets**. A combination of statistical visualization, exploratory analysis, and **unsupervised machine learning** effectively identifies fraudulent transactions. Future enhancements could involve **ensemble models**, hybrid supervised-unsupervised approaches, or real-time anomaly detection to further improve detection accuracy and reduce false positives.

---

