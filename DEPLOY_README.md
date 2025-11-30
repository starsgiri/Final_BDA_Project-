# 🫀 Heart Disease Prediction System

AI-powered heart disease risk assessment using Apache Spark ML and Google Gemini AI.

## 🚀 Quick Start (Streamlit Cloud Deployment)

### 1. **Fork this repository**

### 2. **Configure Secrets** (IMPORTANT!)

When deploying on Streamlit Cloud:

1. Go to your app settings
2. Click on "Secrets"
3. Add the following:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. **Deploy**

- Main file: `web-apps/Streamlet_app1.py`
- Python version: 3.10+
- Requirements: Automatically installed from `requirements.txt`

## 📋 Features

- ✅ Real-time heart disease risk prediction
- ✅ AI-powered personalized health suggestions
- ✅ Interactive web interface
- ✅ BMI calculator and risk assessment
- ✅ Detailed health recommendations

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
cd web-apps
streamlit run Streamlet_app1.py
```

### Configure API Key Locally

Create `web-apps/.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

## 📊 Model Information

- **Algorithm**: Gradient Boosted Trees (GBT) Classifier
- **Framework**: Apache Spark ML
- **AI Model**: Google Gemini 2.5 Flash
- **Accuracy**: 79%

## 🛡️ Security

- API keys are stored securely in Streamlit secrets
- Never commit `.streamlit/secrets.toml` to Git
- Secrets file is in `.gitignore`

## 📝 Usage

1. Enter patient information (age, BMI, lifestyle factors)
2. Click "Run Prediction"
3. View risk assessment
4. Get AI-powered personalized health suggestions

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. Always consult healthcare professionals for medical advice.

## 📄 License

Educational use only.

---

Built with ❤️ using Streamlit, Apache Spark ML, and Google Gemini AI
