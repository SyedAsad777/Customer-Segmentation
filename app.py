import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load trained K-Means model and scaler
try:
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Function to generate business recommendations
def generate_recommendations(cluster_summary):
    recommendations = []

    if cluster_summary["High-Value Customers (%)"] > 50:
        recommendations.append("Focus on retaining high-value customers with loyalty programs and exclusive discounts.")
    elif cluster_summary["High-Value Customers (%)"] < 20:
        recommendations.append("Increase high-value customer base through personalized marketing campaigns and premium offerings.")

    if cluster_summary["Medium-Value Customers (%)"] > 40:
        recommendations.append("Convert medium-value customers into high-value customers by offering bundle deals and incentives.")

    if cluster_summary["Low-Value Customers (%)"] > 30:
        recommendations.append("Engage low-value customers with targeted promotions, first-time buyer discounts, and re-engagement emails.")

    if not recommendations:
        recommendations.append("Customer segmentation is balanced. Continue monitoring trends and optimize customer engagement strategies.")

    return recommendations

# Upload and process file
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Standardizing column names
        df.columns = df.columns.str.strip().str.lower()

        # Convert InvoiceDate to datetime
        if "invoicedate" in df.columns:
            df["invoicedate"] = pd.to_datetime(df["invoicedate"], format="%d-%m-%Y %H:%M", errors='coerce')

        # Remove cancelled transactions (InvoiceNo starts with 'C')
        df = df[~df["invoiceno"].astype(str).str.startswith("C")]

        # Remove missing CustomerIDs
        df = df.dropna(subset=["customerid"])

        # Convert CustomerID to integer
        df["customerid"] = df["customerid"].astype(int)

        # Create TotalPrice feature
        df["totalprice"] = df["quantity"] * df["unitprice"]

        # Remove negative quantities and unit prices
        df = df[(df["quantity"] > 0) & (df["unitprice"] > 0)]

        # Compute Recency, Frequency, and Monetary (RFM)
        latest_date = df["invoicedate"].max()
        rfm = df.groupby("customerid").agg({
            "invoicedate": lambda x: (latest_date - x.max()).days,  # Recency
            "invoiceno": "count",  # Frequency
            "totalprice": "sum"  # Monetary
        }).rename(columns={"invoicedate": "Recency", "invoiceno": "Frequency", "totalprice": "Monetary"})

        # Standardize RFM features
        rfm_scaled = scaler.transform(rfm)

        # Predict customer segments
        rfm["Cluster"] = kmeans.predict(rfm_scaled)

        # Customer segmentation percentages
        cluster_counts = rfm["Cluster"].value_counts(normalize=True) * 100
        cluster_summary = {
            "Low-Value Customers (%)": round(cluster_counts.get(0, 0), 2),
            "Medium-Value Customers (%)": round(cluster_counts.get(1, 0), 2),
            "High-Value Customers (%)": round(cluster_counts.get(2, 0), 2),
        }

        # Get top 5 and bottom 5 products
        product_sales = df.groupby("description")["quantity"].sum().sort_values(ascending=False)
        top_5_products = product_sales.head(5).to_dict()
        bottom_5_products = product_sales[product_sales > 0].tail(5).to_dict()

        # Generate Business Recommendations
        business_recommendations = generate_recommendations(cluster_summary)

        result = {
            "status": "success",
            "cluster_summary": cluster_summary,
            "top_5_products": top_5_products,
            "bottom_5_products": bottom_5_products,
            "business_recommendations": business_recommendations
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)





