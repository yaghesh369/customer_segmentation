import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load your customer data with income in Lakhs and spending score
df = pd.read_csv('customers.csv')

# Use consistent column names matching your app:
features = df[['Age', 'Annual Income (Lakhs)', 'Spending Score']]

# Train KMeans with 3 clusters
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(features)  # Assign clusters to each record

# Save the trained model to segmentation folder
joblib.dump(model, 'segmentation/kmeans_model.joblib')

# Save clustered data for app use
df.to_csv('segmentation/customers_clustered.csv', index=False)

print('Model trained and data saved with cluster labels.')
