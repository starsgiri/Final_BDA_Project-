# Gemini API Setup Guide

## Overview
Both Streamlit apps (`app.py` and `Streamlet_app1.py`) use Google's Gemini AI to provide personalized health suggestions based on the heart disease prediction results.

## Getting Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key (keep it secure!)

## Configuration Options

### Option 1: Using Streamlit Secrets (Recommended)

1. Open the file: `.streamlit/secrets.toml`
2. Replace `your-gemini-api-key-here` with your actual API key:
   ```toml
   GEMINI_API_KEY = "AIzaSyBljt2iRev186_7a21AFeY9XJskLXM17a0"
   ```
3. Save the file
4. Restart your Streamlit app

**Note:** This file is already added to `.gitignore` to prevent accidental commits.

### Option 2: Using Environment Variable (Alternative)

For Linux/Mac:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

For Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### Option 3: Manual Input (app.py only)

If you don't configure the API key in secrets, `app.py` will show a text input in the sidebar where you can enter your key manually each time you run the app.

## Features Enabled by Gemini AI

### In `Streamlet_app1.py`:
- **Personalized Health Plan**: AI analyzes all patient data and provides:
  - Medical follow-up recommendations
  - Diet & nutrition advice
  - Exercise & physical activity suggestions
  - Lifestyle modifications
  - Mental health & stress management tips

### In `app.py`:
- **AI-Powered Health Insights**: Provides:
  - Risk level interpretation
  - Top 3 actionable lifestyle changes
  - Positive encouragement
  - Healthcare consultation recommendations

## Running the Apps

### Run Streamlet_app1.py:
```bash
cd web-apps
streamlit run Streamlet_app1.py
```

### Run app.py:
```bash
cd web-apps
streamlit run app.py
```

## Troubleshooting

### "Gemini AI Not Configured" Warning
- Ensure you've added your API key to `.streamlit/secrets.toml`
- Check that the file is in the correct location
- Verify the API key is valid and hasn't expired

### API Rate Limits
- Free tier: 60 requests per minute
- If you hit limits, wait a minute before trying again
- Consider upgrading to paid tier for production use

### API Errors
- Check your API key is correct
- Ensure you have internet connectivity
- Verify the API key has the necessary permissions

## Security Best Practices

✅ **DO:**
- Store API keys in `.streamlit/secrets.toml`
- Add secrets file to `.gitignore`
- Use environment variables for production
- Rotate keys regularly

❌ **DON'T:**
- Commit API keys to Git
- Share keys publicly
- Hardcode keys in source files
- Use the same key across multiple projects

## Sample Output

When configured correctly, after making a prediction, you'll see:
- Standard risk assessment
- Traditional recommendations
- **NEW:** AI-generated personalized health plan with detailed, context-aware suggestions

## Links

- [Google AI Studio](https://makersuite.google.com/app/apikey)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Streamlit Secrets Documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)

## Support

For issues or questions:
1. Check the Gemini API status: https://status.cloud.google.com/
2. Review Streamlit logs for error messages
3. Verify API key configuration

---

**Last Updated:** November 30, 2025
