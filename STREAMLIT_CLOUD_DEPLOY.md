# 🚀 Streamlit Cloud Deployment Guide

## Step-by-Step Deployment Instructions

### 1. **Push to GitHub**

```bash
git push origin main
```

### 2. **Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `Final_BDA_Project-`
5. Set main file path: `web-apps/Streamlet_app1.py`
6. Click "Advanced settings"

### 3. **Configure API Key (CRITICAL!)**

In the "Secrets" section, paste:

```toml
GEMINI_API_KEY = "AIzaSyD9LLkzRJsVriT4nS-kIsnX96S8dtE-SJI"
```

**⚠️ Important Notes:**
- This secret will be encrypted and securely stored
- Never share your Streamlit Cloud URL publicly if using free API tier
- The secret is NOT visible in your GitHub repository
- You can rotate the key anytime in app settings

### 4. **Optional: Advanced Settings**

- **Python version**: 3.10 or higher
- **Timezone**: Your timezone
- **Resources**: Default (free tier)

### 5. **Deploy!**

Click "Deploy" button. The app will:
1. Install dependencies from `requirements.txt`
2. Load the PySpark ML model
3. Initialize Gemini AI with your secret key
4. Start the web interface

### 6. **Verify Deployment**

Once deployed, test:
1. Fill out patient information
2. Click "Run Prediction"
3. Check if AI suggestions appear (this confirms Gemini is working)

---

## 📋 App Configuration Summary

| Setting | Value |
|---------|-------|
| **Repository** | `starsgiri/Final_BDA_Project-` |
| **Branch** | `main` |
| **Main file** | `web-apps/Streamlet_app1.py` |
| **Python version** | 3.10+ |
| **API Key** | Stored in Streamlit Secrets |

---

## 🔧 Troubleshooting

### Issue: "Model not found"
- Ensure the `models/model1(79)/model1` directory is in your repository
- Check that model files were committed

### Issue: "Gemini AI not working"
- Verify API key is correctly added to Secrets
- Check format: `GEMINI_API_KEY = "your-key"`
- Ensure no extra spaces or quotes

### Issue: "Dependencies failed to install"
- Check `requirements.txt` is in root directory
- Verify all package names are correct

### Issue: "App crashes on startup"
- Check Streamlit Cloud logs
- Verify Spark model path is correct
- Ensure all required files are committed

---

## 📱 Sharing Your App

Once deployed, you'll get a URL like:
```
https://your-username-final-bda-project.streamlit.app
```

Share this with:
- ✅ Healthcare professionals for feedback
- ✅ Academic reviewers
- ✅ Project collaborators

⚠️ **Free Tier Limits:**
- Gemini API: 15 requests/minute
- Streamlit: Public apps may sleep after inactivity
- Consider upgrading for production use

---

## 🔄 Updating Your App

To update after deployment:

```bash
# Make changes to your code
git add web-apps/Streamlet_app1.py
git commit -m "Update: describe changes"
git push origin main
```

Streamlit Cloud will automatically redeploy!

---

## 🛡️ Security Checklist

Before deploying:
- [x] API key stored in Streamlit Secrets (not in code)
- [x] `.streamlit/secrets.toml` in `.gitignore`
- [x] No hardcoded sensitive data
- [x] Proper error handling for API calls
- [x] User disclaimers in place

---

## 📊 Monitoring

After deployment, monitor:
- App analytics (Streamlit Cloud dashboard)
- API usage (Google AI Studio)
- Error logs (Streamlit Cloud logs)
- User feedback

---

**Your app is now live! 🎉**

URL will be: `https://[your-username]-final-bda-project.streamlit.app`
