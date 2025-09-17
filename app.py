import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("processed", exist_ok=True)


df = pd.read_csv("processed/data_with_clusters.csv")
X = df[["x", "y"]]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_path = "processed/kmeans_model.pkl"
if not os.path.exists(kmeans_path):
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
    joblib.dump(kmeans, kmeans_path)
else:
    kmeans = joblib.load(kmeans_path)


cluster_to_class = {0: "🟢 Class 1", 1: "🔵 Class 2", 2: "🔴 Class 0"}


def predict_and_plot(x, y):
    scaled = scaler.transform([[x, y]])
    cluster_id = kmeans.predict(scaled)[0]
    class_name = cluster_to_class[cluster_id]

    
    plt.figure(figsize=(6, 5))
    plt.scatter(X["x"], X["y"], c=kmeans.labels_, cmap="Set1", alpha=0.6, s=40)
    plt.scatter(x, y, c="black", s=200, marker="*", edgecolors="white", label="Your Point")
    plt.legend()
    plt.title(f"Cluster {cluster_id} → {class_name}")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")

    plot_path = "outputs/prediction_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return f"✅ Predicted: Cluster {cluster_id} → {class_name}", plot_path

with gr.Blocks(css="""
    body {background: linear-gradient(135deg, #f9f9f9, #e6f7ff);}
    .gradio-container {font-family: 'Segoe UI', sans-serif;}
    h1 {color: #004080; text-align: center;}
    .prediction-box {font-size: 20px; font-weight: bold; color: #333;}
""") as demo:

    gr.Markdown("## ✨ Cluster Prediction App")
    gr.Markdown("Enter **X** and **Y** values below to see predicted cluster and class assignment.")

    with gr.Row():
        with gr.Column():
            x_in = gr.Number(label="Feature X", value=0.0)
            y_in = gr.Number(label="Feature Y", value=0.0)
            predict_btn = gr.Button("🔮 Predict", variant="primary")

        with gr.Column():
            result = gr.Textbox(label="Prediction Result", elem_classes="prediction-box")
            plot = gr.Image(type="filepath", label="Cluster Visualization")

    predict_btn.click(predict_and_plot, inputs=[x_in, y_in], outputs=[result, plot])

demo.launch()
