# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Step 1: Load the dataset (ensure the dataset is available in your working directory)
# Example dataset name: 'credit_card_data.csv'
# Adjust the dataset path as necessary
df = pd.read_csv('credit_card_data.csv')
# Step 2: Data Preprocessing
# Drop rows with missing values for simplicity (alternatively, handle missing values properly with imputation)
df = df.dropna()
# Select features and target for credit scoring
# Assuming 'default' is the target and 'customer_id' is an irrelevant feature
X = df.drop(['default', 'customer_id'], axis=1) # Adjust column names as per your dataset
y = df['default']
# Step 3: Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Step 4: Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 5: Handle class imbalance using SMOTE (for binary classification)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
# Step 6: Build a Random Forest Classifier for Credit Scoring
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_sm, y_train_sm)
# Predict on the test set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
# Step 7: Evaluate the model
print("Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Plot feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
# Step 8: Customer Segmentation Using K-Means Clustering
# Choose features for segmentation (e.g., income, credit limit, credit utilization, etc.)
segmentation_features = df[['credit_limit', 'income', 'age', 'credit_utilization']] #
Adjust as per dataset
# Standardize the features for clustering
segmentation_features_scaled = scaler.fit_transform(segmentation_features)
# Dimensionality reduction using PCA for visualization
pca = PCA(n_components=2)
segmentation_pca = pca.fit_transform(segmentation_features_scaled)
# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['segment'] = kmeans.fit_predict(segmentation_pca)
# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(segmentation_pca[:, 0], segmentation_pca[:, 1], c=df['segment'], cmap='viridis')
plt.title('Customer Segmentation (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
# Step 9: Analyze and Interpret Segments
# Summarize the characteristics of each segment
segment_summary = df.groupby('segment').mean()
print(segment_summary)
# Pair plot to visualize segment characteristics
sns.pairplot(df, hue='segment', vars=['credit_limit', 'income', 'age', 'credit_utilization'])
plt.show()
