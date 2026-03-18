import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

# Plot size
rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# CSV file path - make sure ye file wahan hai jahan script hai
data = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\creditcard fraud detection\creditcard.csv", sep=',')

# Data check
print(data.head())
print(data.info())
print("Any missing values? ", data.isnull().values.any())

# Transaction class distribution
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Separate fraud and normal data
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print("Fraud Shape:", fraud.shape, "Normal Shape:", normal.shape)

# Amount description
print("Fraud Amount Description:\n", fraud.Amount.describe())
print("Normal Amount Description:\n", normal.Amount.describe())

# Plot amount per transaction by class
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins=bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()

# Time vs Amount plot - fixed variable names
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# Take sample of data
data1 = data.sample(frac=0.1, random_state=1)
print("Sample shape:", data1.shape)
print("Original shape:", data.shape)

# Fraud and valid in sample
Fraud = data1[data1['Class'] == 1]
Valid = data1[data1['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print("Outlier Fraction:", outlier_fraction)
print("Fraud Cases:", len(Fraud))
print("Valid Cases:", len(Valid))

# Correlation
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
sns.heatmap(data1[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Features and target
columns = [c for c in data1.columns if c != "Class"]
target = "Class"
state = np.random.RandomState(42)

X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X), 
                                        contamination=outlier_fraction, random_state=state, verbose=0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine": OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.05,
                                          max_iter=-1)
}

n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Reshape predictions to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    # Classification metrics
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score:")
    print(accuracy_score(Y, y_pred))
    print("Classification Report:")
    print(classification_report(Y, y_pred))
