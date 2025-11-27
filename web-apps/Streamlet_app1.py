import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction System", 
    page_icon="❤️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI/UX ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card-like containers */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics enhancement */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Spark Session Setup ---
@st.cache_resource
def get_spark_session():
    """Creates and caches the Spark Session"""
    spark = SparkSession.builder \
        .appName("HeartDiseasePredictionApp") \
        .master("local[*]") \
        .getOrCreate()
    return spark

spark = get_spark_session()

# --- 3. Model Loading ---
@st.cache_resource
def load_model(_spark):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "models", "model1(79)", "model1")
    model_path = os.path.normpath(model_path)
    
    try:
        model = PipelineModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from path: {model_path}")
        st.error(f"Details: {e}")
        return None

model = load_model(spark)

# --- 4. Mappings ---
age_map = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80 or older": 13
}

gen_health_map = {
    "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
}

# BMI Categories
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif 18.5 <= bmi < 25:
        return "Normal Weight", "🟢"
    elif 25 <= bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

# Risk assessment function
def assess_risk_level(probability, prediction):
    """Enhanced risk assessment with multiple factors"""
    risk_prob = probability[1] * 100
    
    if risk_prob >= 80.0:
        return "HIGH RISK", "🔴", "#f44336", risk_prob
    elif risk_prob >= 60.0:
        return "MODERATE RISK", "🟡", "#ff9800", risk_prob
    elif risk_prob >= 40.0:
        return "LOW-MODERATE RISK", "🟠", "#ffc107", risk_prob
    else:
        return "LOW RISK", "🟢", "#4caf50", risk_prob

# --- 5. Sidebar Information ---
with st.sidebar:
    st.markdown("## 📊 About This Tool")
    st.markdown("""
    This AI-powered system uses **Apache Spark ML** to predict heart disease risk based on:
    
    -  **Demographics**: Age, Sex, BMI
    -  **Medical History**: BP, Cholesterol, Diabetes
    -  **Lifestyle Factors**: Activity, Smoking
    -  **Health Metrics**: Physical & Mental Health
    """)
    
    st.markdown("---")
    st.markdown("###  Model Information")
    if model:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model Not Loaded")
    
    st.markdown("---")
    st.markdown("### 📈 Risk Categories")
    st.markdown("""
    - 🟢 **Low Risk**: < 40%
    - 🟠 **Low-Moderate**: 40-60%
    - 🟡 **Moderate Risk**: 60-80%
    - 🔴 **High Risk**: > 80%
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: All predictions should be reviewed by healthcare professionals")

# --- 6. Main Header ---
st.markdown("""
<div class="custom-header">
    <h1>❤️ Heart Disease Risk Prediction System</h1>
    <p>Advanced ML-powered health risk assessment tool</p>
</div>
""", unsafe_allow_html=True)

# --- 7. User Interface ---
with st.form("prediction_form"):
    st.markdown("###  Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("####  Demographics")
        age_input = st.selectbox("Age Group", options=list(age_map.keys()), index=6)
        sex_input = st.radio("Sex", ["Male", "Female"], horizontal=True)
        height_cm = st.number_input("Height (cm)", 100, 250, 175)
        weight_kg = st.number_input("Weight (kg)", 30, 200, 80)
        
    with col2:
        st.markdown("####  Medical History")
        bp_input = st.selectbox("High Blood Pressure?", ["No", "Yes"])
        chol_input = st.selectbox("High Cholesterol?", ["No", "Yes"])
        stroke_input = st.selectbox("History of Stroke?", ["No", "Yes"])
        diab_input = st.selectbox("Has Diabetes?", ["No", "Yes"])
        
    with col3:
        st.markdown("####  Lifestyle & Health")
        walk_input = st.radio("Difficulty Walking/Climbing Stairs?", ["No", "Yes"], horizontal=True)
        smoke_input = st.radio("Smoked >100 cigarettes (lifetime)?", ["No", "Yes"], horizontal=True)
        phys_act = st.radio("Physical Activity (past 30 days)?", ["No", "Yes"], horizontal=True)
        gen_hlth = st.select_slider("General Health Rating", options=list(gen_health_map.keys()), value="Good")
    
    st.markdown("---")
    col4, col5 = st.columns(2)
    
    with col4:
        phys_days = st.slider("Days physical health was bad (past 30 days)", 0, 30, 0)
    
    with col5:
        ment_days = st.slider("Days mental health was bad (past 30 days)", 0, 30, 0)
    
    st.markdown("---")
    submit_btn = st.form_submit_button("🔍 Run Prediction", type="primary", use_container_width=True)

# --- 8. Prediction Logic ---
if submit_btn and model:
    # Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    bmi_category, bmi_icon = get_bmi_category(bmi)
    
    # Prepare Data
    input_data = [{
        'BMI': bmi,
        'Smoking': 1 if smoke_input == "Yes" else 0,
        'Stroke': 1 if stroke_input == "Yes" else 0,
        'Diabetes': 1 if diab_input == "Yes" else 0,
        'PhysicalActivity': 1 if phys_act == "Yes" else 0,
        'GenHealth': gen_health_map[gen_hlth],
        'PhysHealthDays': phys_days,
        'MentHealthDays': ment_days,
        'Male': 1 if sex_input == "Male" else 0,
        'AgeCategory': age_map[age_input],
        'HighBP': 1 if bp_input == "Yes" else 0,
        'HighChol': 1 if chol_input == "Yes" else 0,
        'DiffWalk': 1 if walk_input == "Yes" else 0
    }]
    
    # Create Spark DataFrame
    input_df = spark.createDataFrame(input_data)
    
    # Run Prediction with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(" Initializing prediction...")
        progress_bar.progress(25)
        time.sleep(0.3)
        
        status_text.text(" Running ML model...")
        prediction = model.transform(input_df)
        progress_bar.progress(75)
        time.sleep(0.3)
        
        status_text.text(" Analyzing results...")
        result = prediction.select("prediction", "probability").collect()[0]
        progress_bar.progress(100)
        time.sleep(0.3)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Extract results
        pred_label = result['prediction']
        probs = result['probability']
        
        # Assess risk
        risk_level, risk_icon, risk_color, risk_percentage = assess_risk_level(probs, pred_label)
        
        # Display Results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Top metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("BMI", f"{bmi:.1f}", f"{bmi_icon} {bmi_category}")
        
        with metric_col2:
            st.metric("Risk Probability", f"{risk_percentage:.1f}%")
        
        with metric_col3:
            st.metric("Prediction", f"{pred_label:.0f}")
        
        with metric_col4:
            st.metric("Confidence", f"{max(probs[0], probs[1]):.1%}")
        
        st.markdown("---")
        
        # Main result card
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}11 100%); border-left: 5px solid {risk_color};">
                <h2>{risk_icon} {risk_level}</h2>
                <h3>Risk Probability: {risk_percentage:.2f}%</h3>
                <p style="font-size: 1.1em; margin-top: 1rem;">
                    Based on the provided health metrics, the model indicates <strong>{risk_level.lower()}</strong> 
                    of heart disease indicators.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(risk_percentage / 100)
            
            # Risk factors summary
            st.markdown("###  Key Risk Factors Identified")
            risk_factors = []
            
            if input_data[0]['HighBP'] == 1:
                risk_factors.append("🔴 High Blood Pressure")
            if input_data[0]['HighChol'] == 1:
                risk_factors.append("🔴 High Cholesterol")
            if input_data[0]['Smoking'] == 1:
                risk_factors.append("🔴 Smoking History")
            if input_data[0]['Diabetes'] == 1:
                risk_factors.append("🔴 Diabetes")
            if input_data[0]['Stroke'] == 1:
                risk_factors.append("🔴 Previous Stroke")
            if bmi >= 30:
                risk_factors.append("🔴 Obesity (BMI ≥ 30)")
            if input_data[0]['PhysicalActivity'] == 0:
                risk_factors.append("🟡 Lack of Physical Activity")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.success(" No major risk factors identified")
        
        with result_col2:
            st.markdown("### 💊 Recommendations")
            
            recommendations = []
            
            if risk_percentage >= 60:
                recommendations.append("🏥 Consult a cardiologist immediately")
                recommendations.append("📋 Schedule comprehensive health screening")
            
            if input_data[0]['HighBP'] == 1 or input_data[0]['HighChol'] == 1:
                recommendations.append("💊 Monitor BP and cholesterol regularly")
            
            if input_data[0]['Smoking'] == 1:
                recommendations.append("🚭 Consider smoking cessation programs")
            
            if input_data[0]['PhysicalActivity'] == 0:
                recommendations.append("🏃 Start regular physical activity")
            
            if bmi >= 25:
                recommendations.append("🥗 Adopt a heart-healthy diet")
            
            if phys_days > 10 or ment_days > 10:
                recommendations.append("🧘 Focus on mental and physical wellness")
            
            if not recommendations:
                recommendations.append("✅ Maintain current healthy lifestyle")
                recommendations.append("📅 Regular health check-ups")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Detailed probability breakdown
        with st.expander("📈 View Detailed Probability Breakdown"):
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric("Probability of NO Heart Disease", f"{probs[0]:.2%}")
                st.progress(float(probs[0]))
            
            with prob_col2:
                st.metric("Probability of Heart Disease", f"{probs[1]:.2%}")
                st.progress(float(probs[1]))
        
        # Raw input data
        with st.expander("🔍 View Raw Input Data"):
            st.json(input_data[0])
        
        # Disclaimer
        st.warning(" **Medical Disclaimer**: This prediction is for informational purposes only and should not replace professional medical advice. Please consult with healthcare providers for proper diagnosis and treatment.")
        
    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.error(f"Error details: {e}")
        
elif submit_btn and not model:
    st.error("❌ Model is not loaded. Please check the model path and try again.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit and Apache Spark ML</p>
    <p><small>© 2024 Heart Disease Prediction System | For Educational Purposes</small></p>
</div>
""", unsafe_allow_html=True)
