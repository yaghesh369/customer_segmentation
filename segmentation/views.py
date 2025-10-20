from django.shortcuts import render
import joblib
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
import os
from django.conf import settings

model = joblib.load('segmentation/kmeans_model.joblib')

CLUSTER_DESCRIPTIONS = {
    0: "Cluster 0: Young customers with low to moderate income and variable spending.",
    1: "Cluster 1: Middle-aged customers with moderate income and higher spending patterns.",
    2: "Cluster 2: Older customers with higher income but lower spending scores.",
}

def home(request):
    cluster = None
    cluster_desc = None
    age = lakhs = score = None
    batch_results = None
    df_for_plot = None
    input_point = None

    if request.method == 'POST':
        if 'datafile' in request.FILES:
            try:
                df_uploaded = pd.read_csv(request.FILES['datafile'])
                df_filtered = df_uploaded[df_uploaded['Age'] > 18]

                # Rename to model expected columns
                df_filtered.rename(columns={
                    'Lakhs': 'Annual Income (Lakhs)',
                    'Spending_Score': 'Spending Score'
                }, inplace=True)

                required_cols = ['Age', 'Annual Income (Lakhs)', 'Spending Score']
                if all(col in df_filtered.columns for col in required_cols):
                    features = df_filtered[required_cols]
                    df_filtered['Cluster'] = model.predict(features)
                    batch_results = df_filtered.copy()

                    batch_results.rename(columns={
                        'Annual Income (Lakhs)': 'Lakhs',
                        'Spending Score': 'Spending_Score'
                    }, inplace=True)
                    batch_results = batch_results.to_dict('records')
                    df_for_plot = df_filtered
                else:
                    batch_results = 'CSV file must contain columns: Age, Lakhs, Spending_Score'
            except Exception as e:
                batch_results = f'Error processing CSV file: {e}'

        else:
            try:
                age = float(request.POST.get('age'))
                lakhs = float(request.POST.get('lakhs'))
                score = float(request.POST.get('score'))
                cluster = model.predict([[age, lakhs, score]])[0]
                cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster, "No description available.")

                input_point = {'Age': age, 'Annual Income (Lakhs)': lakhs, 'Spending Score': score, 'Cluster': cluster}

                data_path = os.path.join(settings.BASE_DIR, 'segmentation/customers_clustered.csv')
                if os.path.exists(data_path):
                    df_for_plot = pd.read_csv(data_path)
                else:
                    df_for_plot = pd.DataFrame(columns=['Age', 'Annual Income (Lakhs)', 'Spending Score', 'Cluster'])

                df_for_plot = pd.concat([df_for_plot, pd.DataFrame([input_point])], ignore_index=True)

            except Exception:
                cluster = 'Invalid input'

    else:
        data_path = os.path.join(settings.BASE_DIR, 'segmentation/customers_clustered.csv')
        if os.path.exists(data_path):
            df_for_plot = pd.read_csv(data_path)
        else:
            df_for_plot = None

    plot_url = None
    if df_for_plot is not None and not df_for_plot.empty:
        plt.figure(figsize=(9,6))
        colors = ['red', 'blue', 'green']
        df_for_plot_plot = df_for_plot.rename(columns={
            'Annual Income (Lakhs)': 'Lakhs',
            'Spending Score': 'Spending_Score'
        })

        for c in df_for_plot_plot['Cluster'].unique():
            subset = df_for_plot_plot[df_for_plot_plot['Cluster'] == c]
            plt.scatter(subset['Lakhs'], subset['Spending_Score'], c=colors[c % len(colors)], label=f'Cluster {c}', s=60, alpha=0.7)

        if input_point:
            plt.scatter([input_point['Annual Income (Lakhs)']], [input_point['Spending Score']], c='black', s=150, marker='X', label='Your Input')
            plt.annotate(f"Age: {age}\nCluster: {cluster}",
                         (input_point['Annual Income (Lakhs)'], input_point['Spending Score']),
                         textcoords="offset points", xytext=(15,-30), ha='left',
                         bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.75),
                         fontsize=10)

        plt.xlabel('Annual Income (Lakhs)', fontsize=12)
        plt.ylabel('Spending Score', fontsize=12)
        plt.title('Customer Clusters', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plot_path = os.path.join(settings.BASE_DIR, 'segmentation/static/segmentation/cluster_plot.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        plot_url = 'segmentation/cluster_plot.png'

    table_data = []
    default_csv_path = os.path.join(settings.BASE_DIR, 'segmentation/customers_clustered.csv')
    if os.path.exists(default_csv_path):
        df_table = pd.read_csv(default_csv_path)
        df_table.rename(columns={
            'Annual Income (Lakhs)': 'Lakhs',
            'Spending Score': 'Spending_Score'
        }, inplace=True)
        table_data = df_table.head(10).to_dict('records')

    return render(request, 'segmentation/index.html', {
        'cluster': cluster,
        'cluster_desc': cluster_desc,
        'batch_results': batch_results,
        'plot_url': plot_url,
        'table_data': table_data,
        'age': age,
        'lakhs': lakhs,
        'score': score,
    })
