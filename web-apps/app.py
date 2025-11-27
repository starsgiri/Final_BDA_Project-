import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
import findspark
import google.generativeai as genai
import time

# 1. Initialize Spark (Only once)
findspark.init()

@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("HeartDiseaseApp") \
        .master("local[*]") \
        .getOrCreate()

@st.cache_resource
def load_model():
    model_path = "/home/giri/Desktop/Final_BDA_Project/Final_BDA_Project-/models/model2/v2_dataset"
    return PipelineModel.load(model_path)

spark = get_spark_session()
model = load_model()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioAI - Heart Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR AMAZING UI ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container Animation */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Card Styles */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 20px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Animated Header */
    .main-header {
        text-align: center;
        padding: 40px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(8, 200, 150, 0.9);
        font-size: 1.2rem;
        margin-top: 10px;
    }
    
    /* Risk Score Display */
    .risk-display {
        text-align: center;
        padding: 40px;
        border-radius: 20px;
        margin: 20px 0;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .risk-low {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .risk-score {
        font-size: 4rem;
        font-weight: 700;
        margin: 20px 0;
        animation: scaleIn 0.5s ease-out;
    }
    
    @keyframes scaleIn {
        from { transform: scale(0); }
        to { transform: scale(1); }
    }
    
    /* Recommendation Cards */
    .recommendation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: white;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0;
            transform: translateX(-20px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 50px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Loading Animation */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }
    
    .heart-loader {
        font-size: 3rem;
        animation: heartbeat 1.5s ease-in-out infinite;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        25% { transform: scale(1.1); }
        50% { transform: scale(1); }
    }
    
    /* AI Response Box */
    .ai-response {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border-left: 5px solid #667eea;
        animation: fadeIn 1s ease-in;
    }
    
    .ai-response h3 {
        color: #667eea;
        margin-bottom: 15px;
    }
    
    /* Progress Bar */
    .progress-bar {
        width: 100%;
        height: 30px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        overflow: hidden;
        margin: 20px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 1s ease-out;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -100% 0; }
        100% { background-position: 100% 0; }
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- ANIMATED HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🫀 CardioAI</h1>
    <p>AI-Powered Cardiac Health Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# --- GEMINI API CONFIGURATION ---
st.sidebar.markdown("### 🤖 AI Configuration")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get your API key from Google AI Studio")

if gemini_api_key:
    try:
        genai.configure(api_key=AIzaSyBljt2iRev186_7a21AFeY9XJskLXM17a0)
        st.sidebar.success("API key present")
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {str(e)}")

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("---")
st.sidebar.header("📋 Patient Data Profile")

def user_input_features():
    st.sidebar.markdown("#### 👤 Demographics")
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    
    st.sidebar.markdown("#### 💉 Vitals")
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
    high_bp = st.sidebar.selectbox("High Blood Pressure?", ["No", "Yes", "Borderline"])
    high_chol = st.sidebar.selectbox("High Cholesterol?", ["No", "Yes"])
    
    st.sidebar.markdown("#### 🚬 Lifestyle Habits")
    smoker = st.sidebar.selectbox("Smoked at least 100 cigarettes in life?", ["No", "Yes"])
    alcohol = st.sidebar.slider("Alcohol Consumption (Days per month)", 0, 30, 0)
    phys_activity = st.sidebar.selectbox("Physical Activity in past 30 days?", ["No", "Yes"])
    
    st.sidebar.markdown("#### 🥗 Diet")
    fruit = st.sidebar.slider("Fruit Consumption (Times per day)", 0, 5, 1)
    veggies = st.sidebar.slider("Vegetable Consumption (Times per day)", 0, 5, 1)
    
    st.sidebar.markdown("#### 🏥 Medical History")
    stroke = st.sidebar.selectbox("History of Stroke?", ["No", "Yes"])
    diabetes = st.sidebar.selectbox("History of Diabetes?", ["No", "Yes", "Borderline"])
    gen_health = st.sidebar.select_slider("General Health Rating", 
                                          options=["Excellent", "Very Good", "Good", "Fair", "Poor"])
    
    st.sidebar.markdown("#### 🧠 Wellbeing")
    ment_health = st.sidebar.slider("Days with Poor Mental Health (past 30 days)", 0, 30, 0)
    phys_health = st.sidebar.slider("Days with Poor Physical Health (past 30 days)", 0, 30, 0)

    # --- MAPPING INPUTS TO MODEL FORMAT ---
    data = {
        'HighBP': 1 if high_bp == "Yes" else (3 if high_bp == "No" else 4),
        'HighChol': 1 if high_chol == "Yes" else 2,
        'BMI': bmi,
        'Smoker': 1 if smoker == "Yes" else 2,
        'Stroke': 1 if stroke == "Yes" else 2,
        'Diabetes': 1 if diabetes == "Yes" else (3 if diabetes == "No" else 4),
        'PhysActivity': 1 if phys_activity == "Yes" else 2,
        'Fruits': 100 + fruit if fruit > 0 else 300, 
        'Veggies': 100 + veggies if veggies > 0 else 300,
        'AlcoholCons': 100 + alcohol if alcohol > 0 else 888,
        'GenHealth': ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(gen_health) + 1,
        'MentalHealth': ment_health,
        'PhysHealth': phys_health,
        'Sex': 1 if sex == "Male" else 2,
        'Age': 1
    }
    
    # Age category mapping
    if age < 24: data['Age'] = 1
    elif age < 30: data['Age'] = 2
    elif age < 35: data['Age'] = 3
    elif age < 40: data['Age'] = 4
    elif age < 45: data['Age'] = 5
    elif age < 50: data['Age'] = 6
    elif age < 55: data['Age'] = 7
    elif age < 60: data['Age'] = 8
    elif age < 65: data['Age'] = 9
    elif age < 70: data['Age'] = 10
    elif age < 75: data['Age'] = 11
    elif age < 80: data['Age'] = 12
    else: data['Age'] = 13

    # Store raw values for display
    data['_display'] = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'high_bp': high_bp,
        'high_chol': high_chol,
        'smoker': smoker,
        'alcohol': alcohol,
        'phys_activity': phys_activity,
        'fruit': fruit,
        'veggies': veggies,
        'stroke': stroke,
        'diabetes': diabetes,
        'gen_health': gen_health,
        'ment_health': ment_health,
        'phys_health': phys_health
    }
    
    return data

input_data = user_input_features()

# --- CENTER ANALYSIS BUTTON ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Heart Risk", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if analyze_button:
    # Loading Animation
    with st.spinner(''):
        st.markdown('<div class="loading"><div class="heart-loader">Loading......</div></div>', unsafe_allow_html=True)
        time.sleep(1.5)
    
    # Convert dictionary to Spark DataFrame
    display_data = input_data.pop('_display')
    input_df = spark.createDataFrame([input_data])
    
    # Run Prediction
    prediction = model.transform(input_df)
    
    # Extract results
    probs = prediction.select("probability").collect()[0][0]
    risk_score = probs[1] * 100
    
    # --- DISPLAY RESULTS WITH ANIMATIONS ---
    st.markdown("### 📊 Analysis Results")
    
    # Risk Score Display
    if risk_score < 20:
        risk_class = "risk-low"
        risk_label = "Low Risk"
        emoji = ""
    elif risk_score < 50:
        risk_class = "risk-moderate"
        risk_label = "Moderate Risk"
        emoji = ""
    else:
        risk_class = "risk-high"
        risk_label = "High Risk"
        emoji = ""
    
    st.markdown(f"""
    <div class="risk-display {risk_class}">
        <h2>{emoji} {risk_label}</h2>
        <div class="risk-score">{risk_score:.2f}%</div>
        <p>Probability of Heart Disease</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {risk_score}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- KEY METRICS ---
    st.markdown("### 📈 Key Health Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{display_data['bmi']:.1f}</div>
            <div class="metric-label">BMI</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{display_data['age']}</div>
            <div class="metric-label">Age</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        activity_emoji = "🏃" if display_data['phys_activity'] == "Yes" else "🛋️"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{activity_emoji}</div>
            <div class="metric-label">Activity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health_score = ["Poor", "Fair", "Good", "Very Good", "Excellent"].index(display_data['gen_health']) + 1
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{health_score}/5</div>
            <div class="metric-label">General Health</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- BASIC RECOMMENDATIONS ---
    st.markdown("---")
    st.markdown("### 🩺 Personalized Recommendations")
    
    recommendations = []
    
    if input_data['Smoker'] == 1:
        recommendations.append("🚫 **Stop Smoking:** Smoking is a major cause of heart disease. Cessation can lower risk by 50% in one year.")
    
    if input_data['BMI'] > 30:
        recommendations.append("⚖️ **Weight Management:** Your BMI indicates obesity. Aim for a calorie deficit diet and regular cardio.")
    
    if input_data['HighBP'] == 1:
        recommendations.append("🩸 **Manage Blood Pressure:** Reduce sodium intake and monitor BP daily.")
        
    if input_data['Diabetes'] == 1:
        recommendations.append("🍬 **Control Blood Sugar:** Strict glucose control is vital. Avoid processed sugars.")
        
    if input_data['PhysActivity'] == 2:
        recommendations.append("🏃 **Get Moving:** Aim for at least 30 minutes of walking daily.")
        
    if input_data['Veggies'] == 300 or input_data['Fruits'] == 300:
        recommendations.append("🥦 **Dietary Changes:** Increase intake of plant-based foods rich in antioxidants.")

    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f'<div class="recommendation" style="animation-delay: {i*0.1}s;">{rec}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation">🌟 Great job! Your lifestyle choices are currently supporting good heart health. Keep it up!</div>', unsafe_allow_html=True)
    
    # --- GEMINI AI INSIGHTS ---
    if gemini_api_key:
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Health Insights")
        
        with st.spinner('Generating personalized AI insights...'):
            try:
                # Prepare detailed prompt for Gemini
                prompt = f"""
                You are a compassionate and knowledgeable cardiac health advisor. A patient has received a heart disease risk assessment with the following details:
                
                Risk Score: {risk_score:.2f}%
                Risk Category: {risk_label}
                
                Patient Profile:
                - Age: {display_data['age']} years
                - Sex: {display_data['sex']}
                - BMI: {display_data['bmi']:.1f}
                - High Blood Pressure: {display_data['high_bp']}
                - High Cholesterol: {display_data['high_chol']}
                - Smoker: {display_data['smoker']}
                - Alcohol Consumption: {display_data['alcohol']} days/month
                - Physical Activity: {display_data['phys_activity']}
                - Fruit Consumption: {display_data['fruit']} times/day
                - Vegetable Consumption: {display_data['veggies']} times/day
                - History of Stroke: {display_data['stroke']}
                - Diabetes: {display_data['diabetes']}
                - General Health: {display_data['gen_health']}
                - Mental Health Issues: {display_data['ment_health']} days/month
                - Physical Health Issues: {display_data['phys_health']} days/month
                
                Please provide:
                1. A brief interpretation of their risk level (2-3 sentences)
                2. Top 3 specific, actionable lifestyle changes they should prioritize
                3. Positive encouragement and motivation
                4. When they should consider seeing a healthcare professional
                
                Keep the tone supportive, non-alarmist, and empowering. Format with clear sections.
                """
                
                model_ai = genai.GenerativeModel('gemini-pro')
                response = model_ai.generate_content(prompt)
                
                st.markdown(f"""
                <div class="ai-response">
                    <h3>💡 Personalized AI Analysis</h3>
                    {response.text}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    else:
        st.info("💡 Enter your Gemini API Key in the sidebar to unlock AI-powered personalized health insights!")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
    <p>Always consult with a healthcare provider for proper diagnosis and treatment.</p>
    <p style="margin-top: 20px;">Made with ❤️ using PySpark ML & Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)
