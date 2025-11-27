import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️", layout="wide")

# --- 2. Spark Session Setup (THE FIX) ---
@st.cache_resource
def get_spark_session():
    """
    Creates and caches the Spark Session so it doesn't restart 
    every time the user interacts with the app.
    """
    spark = SparkSession.builder \
        .appName("HeartDiseasePredictionApp") \
        .master("local[*]") \
        .getOrCreate()
    return spark

spark = get_spark_session()

# --- 3. Model Loading ---
@st.cache_resource
def load_model(_spark):
    # Use relative path that works both locally and on Streamlit Cloud
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up one level to project root, then to models
    model_path = os.path.join(script_dir, "..", "models", "model1(79)", "model1")
    model_path = os.path.normpath(model_path)  # Clean up the path
    
    try:
        model = PipelineModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from path: {model_path}")
        st.error(f"Details: {e}")
        return None

model = load_model(spark)

# --- 4. Mappings (for UI to Model conversion) ---
age_map = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80 or older": 13
}

gen_health_map = {
    "Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5
}

# --- 5. User Interface ---
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(f"**Model Status:** {'✅ Loaded' if model else '❌ Not Loaded'}")
st.info("Enter the patient's details below to predict the risk of heart disease using your Spark ML model.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        age_input = st.selectbox("Age Group", options=list(age_map.keys()), index=6)
        sex_input = st.radio("Sex", ["Male", "Female"], horizontal=True)
        height_cm = st.number_input("Height (cm)", 100, 250, 175)
        weight_kg = st.number_input("Weight (kg)", 30, 200, 80)
        
    with col2:
        st.subheader("🏥 Medical History")
        bp_input = st.selectbox("High Blood Pressure?", ["No", "Yes"])
        chol_input = st.selectbox("High Cholesterol?", ["No", "Yes"])
        stroke_input = st.selectbox("History of Stroke?", ["No", "Yes"])
        diab_input = st.selectbox("Has Diabetes?", ["No", "Yes"])
        
    with col3:
        st.subheader("🏃 Lifestyle & Health")
        walk_input = st.radio("Difficulty Walking/Climbing Stairs?", ["No", "Yes"], horizontal=True)
        smoke_input = st.radio("Smoked >100 cigarettes (lifetime)?", ["No", "Yes"], horizontal=True)
        phys_act = st.radio("Physical Activity (past 30 days)?", ["No", "Yes"], horizontal=True)
        gen_hlth = st.select_slider("General Health Rating", options=list(gen_health_map.keys()), value="Good")
        
        st.markdown("---")
        phys_days = st.slider("Days physical health was bad (past 30 days)", 0, 30, 0)
        ment_days = st.slider("Days mental health was bad (past 30 days)", 0, 30, 0)

    submit_btn = st.form_submit_button("Run Prediction", type="primary")

# --- 6. Prediction Logic ---
if submit_btn and model:
    # A. Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # B. Prepare Data Dictionary (Must match Training Columns exactly)
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
    
    # C. Create Spark DataFrame
    input_df = spark.createDataFrame(input_data)
    
    # D. Run Transformation (Prediction)
    with st.spinner("Running Spark Model..."):
        try:
            prediction = model.transform(input_df)
            result = prediction.select("prediction", "probability").collect()[0]
            
            # E. Display Results
            pred_label = result['prediction']
            probs = result['probability']
            
            st.divider()
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Calculated BMI", f"{bmi:.1f}")
                
            with col_res2:
                if probs[1] >= 80.0:
                    st.error(f"**Prediction: HIGH RISK**")
                    st.write(f"Confidence: **{probs[1]:.2%}**")
                    st.write(f"pred lable value: **{pred_label:.2}**")
                    st.progress(float(probs[1]))
                    st.warning("The model suggests a high probability of heart disease indicators.")
                elif probs[1] >=60.0 and prob[1] < 80.0:
                    st.error(f" **Prediction: MODERATE RISK**")
                    st.write(f"Confidence: **{probs[1]:.2%}**")
                    st.write(f"pred lable value: **{pred_label:.2}**")
                    st.progress(float(probs[1]))
                    st.warning("The model suggests a Moderate probability of heart disease indicators.")
                else:
                    st.success(f" **Prediction: LOW RISK**")
                    st.write(f"Confidence: **{probs[0]:.2%}**")
                    st.write(f"pred lable value: **{pred_label:.2}**")
                    st.progress(float(probs[0]))
                    st.caption("Thank you......")
                    
            # Debug: Show raw input data if needed
            with st.expander("View Raw Input Data passed to Spark"):
                st.json(input_data[0])
                
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.error(f"Error details: {e}")
