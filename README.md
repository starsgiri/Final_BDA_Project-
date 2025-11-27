Heart Disease Prediction & Analysis System

A Big Data Analytics project that consolidates multiple heart disease datasets using Apache Spark to predict heart disease risk and provide actionable health suggestions.

🔗 Live Application

Click the link below to access the interactive dashboard:
Launch Streamlit App

📱 Scan to Open on Mobile

📖 Project Overview

This system processes large-scale medical data to assist in the early detection of heart disease. It aggregates data from 5 distinct sources (Cleveland, Hungarian, Switzerland, Long Beach VA, etc.) into a unified schema for robust machine learning analysis.

Key Features

Data Integration: Merges 5 separate heart disease datasets using PySpark.

Data Cleaning: Automated removal of duplicates and handling of null values/outliers.

Risk Prediction: Uses Machine Learning to predict the presence of heart disease (Target 0 or 1).

Smart Suggestions: Provides tailored lifestyle recommendations (e.g., "Consult Cardiologist" vs. "Maintain Lifestyle") based on prediction results and key vitals like Cholesterol and Resting Blood Pressure.

🛠️ Technologies Used

Apache Spark (PySpark): For big data processing, schema standardization, and merging datasets.

Streamlit: For the web-based user interface and dashboard.

Python: Core programming language.

Spark MLlib: For building the predictive machine learning models.
