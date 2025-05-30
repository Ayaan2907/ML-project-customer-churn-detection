import streamlit as st
import pandas as pd
import pickle
import numpy as np  # for averaging probabilities
import os
from dotenv import load_dotenv
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# App title with emoji and styling
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='color: #667eea; font-size: 3rem; margin: 0;'>
        📊 Customer Churn Detection
    </h1>
    <p style='font-size: 1.2rem; color: #888; margin: 10px 0;'>
        AI-Powered Banking Customer Retention Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ——— Load data ———
df = pd.read_csv("churn.csv")
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
locations = sorted(df["Geography"].dropna().unique())

# ——— Model loading ———
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model_paths = {
    "Decision Tree":               "models/dt_model.pkl",
    "K-Nearest Neighbors":         "models/knn_model.pkl",
    "Naive Bayes":                 "models/nb_model.pkl",
    "Random Forest":               "models/rf_model.pkl",
    "Support Vector Machine":      "models/svm_model.pkl",
    "XGBoost":                     "models/xgb_model.pkl",
    "XGB w/ Feature Engineering":  "models/xgboost_featureEng_model.pkl",
    "XGB w/ SMOTE":                "models/xgboost_SMOTE_model.pkl",
    "Voting Classifier":            "models/voting_clf.pkl",
}

models = {name: load_model(path) for name, path in model_paths.items()}

# ——— Input preparation ———
def prepare_input(
    credit_score: int,
    location: str,
    gender: str,
    age: int,
    tenure: int,
    balance: float,
    num_products: int,
    has_credit_card: bool,
    is_active_member: bool,
    estimated_salary: float,
    all_locations: list[str],
) -> tuple[pd.DataFrame, dict]:
    input_dict = {
        "CreditScore":     credit_score,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCrCard":       int(has_credit_card),
        "IsActiveMember":  int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Gender_Male":     1 if gender == "Male" else 0,
        "Gender_Female":   1 if gender == "Female" else 0,
    }
    for loc in all_locations:
        input_dict[f"Geography_{loc}"] = 1 if location == loc else 0

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# ——— Prediction logic ———
def make_predictions(input_df: pd.DataFrame) -> dict[str, float]:
    """
    Returns the churn probability (class=1) for each model in model_subset.
    """
    # Select which models to display
    chosen_models = {
        "XGBoost": models["XGBoost"],
        "Random Forest": models["Random Forest"],
        "K-Nearest Neighbors": models["K-Nearest Neighbors"],
        "Decision Tree": models["Decision Tree"],
        "Naive Bayes": models["Naive Bayes"],
        "Support Vector Machine": models["Support Vector Machine"],
        # "XGB w/ Feature Engineering": models["XGB w/ Feature Engineering"],
        # "XGB w/ SMOTE": models["XGB w/ SMOTE"],
        # "Voting Classifier": models["Voting Classifier"]
    }
    return {
        name: model.predict_proba(input_df)[0][1]
        for name, model in chosen_models.items()
    }

def display_predictions(probabilities: dict[str, float]):
    """
    Display predictions with enhanced visualizations using plotly.
    """
    avg_prob = np.mean(list(probabilities.values()))
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Gauge chart for average probability
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability", 'font': {'size': 24}},
            delta = {'reference': 20, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': 'lightgreen'},
                    {'range': [25, 50], 'color': 'yellow'},
                    {'range': [50, 75], 'color': 'orange'},
                    {'range': [75, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk assessment text
        if avg_prob < 0.25:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        elif avg_prob < 0.50:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
        else:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
            
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: rgba(0,0,0,0.1); border-radius: 10px; margin: 10px 0;'>
            <h3 style='color: {risk_color}; margin: 0;'>{risk_level}</h3>
            <p style='font-size: 18px; margin: 5px 0;'>The customer has a <strong>{avg_prob:.1%}</strong> probability of churning.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Bar chart for individual model predictions
        model_names = list(probabilities.keys())
        model_probs = [prob * 100 for prob in probabilities.values()]
        
        fig_bar = go.Figure(data=[
            go.Bar(
                y=model_names,
                x=model_probs,
                orientation='h',
                marker=dict(
                    color=model_probs,
                    colorscale='RdYlGn_r',
                    cmin=0,
                    cmax=100
                ),
                text=[f'{prob:.1f}%' for prob in model_probs],
                textposition='auto'
            )
        ])
        
        fig_bar.update_layout(
            title="Churn Probability by Model",
            title_font_size=20,
            xaxis_title="Probability (%)",
            yaxis_title="Models",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"},
            xaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

# ——— Prediction explanation ———
def explain_prediction(input_dict: dict, avg_probability: float, feature_importance: dict, churned_stats: dict) -> str:
    """
    Generate a comprehensive prompt for explaining ML model predictions using Groq AI.
    
    Args:
        input_dict: Dictionary containing customer features
        avg_probability: Average churn probability from models
        feature_importance: Dictionary of top 10 feature importances
        churned_stats: Summary statistics for churned customers
    
    Returns:
        str: Formatted prompt for Groq AI
    """
    
    # Format customer information
    customer_info = ""
    for key, value in input_dict.items():
        customer_info += f"  - {key}: {value}\n"
    
    # Format feature importance
    feature_importance_text = ""
    for feature, importance in feature_importance.items():
        feature_importance_text += f"  - {feature}: {importance:.4f}\n"
    
    # Format churned customer statistics
    churned_stats_text = ""
    for stat, value in churned_stats.items():
        churned_stats_text += f"  - {stat}: {value}\n"
    
    prompt = f"""
You are an expert data scientist at a bank. You have been asked to explain the prediction of a machine learning model that predicts whether a customer will churn.

The model predicted that this customer has a {avg_probability:.2%} probability of churning.

Customer Information:
{customer_info}

Top 10 Most Important Features for Predicting Churn:
{feature_importance_text}

Summary Statistics for Churned Customers:
{churned_stats_text}

Based on this information, please provide a detailed explanation of:

1. Why the model made this prediction (focus on the most important features)
2. What factors are increasing or decreasing the churn probability for this specific customer
3. What actionable recommendations the bank could take to retain this customer if they are at high risk of churning
4. How this customer compares to the typical profile of churned customers

Please provide your explanation in a clear, business-friendly language that a bank manager could understand and act upon.
"""
    
    return prompt

def get_feature_importance() -> dict:
    """
    Return the top 10 most important features for churn prediction.
    This would typically come from your trained model's feature_importances_.
    """
    # Placeholder feature importance values - replace with actual values from your model
    return {
        "NumOfProducts": 0.323888,
        "IsActiveMember": 0.164146,
        "Age": 0.109550,
        "Geography_Germany": 0.091373,
        "Balance": 0.052786,
        "Geography_France": 0.046463,
        "Gender_Female": 0.045283,
        "Geography_Spain": 0.036855,
        "CreditScore": 0.035005,
        "EstimatedSalary": 0.032655
    }

def get_churned_customer_stats() -> dict:
    """
    Return summary statistics for churned customers.
    This would typically come from your data analysis.
    """
    # Calculate actual statistics from the churn data
    churned_customers = df[df['Exited'] == 1]
    
    return {
        "Average Age": f"{churned_customers['Age'].mean():.1f} years",
        "Average Credit Score": f"{churned_customers['CreditScore'].mean():.0f}",
        "Average Balance": f"${churned_customers['Balance'].mean():,.2f}",
        "Average Tenure": f"{churned_customers['Tenure'].mean():.1f} years",
        "Average Number of Products": f"{churned_customers['NumOfProducts'].mean():.1f}",
        "Percentage with Credit Card": f"{(churned_customers['HasCrCard'].mean() * 100):.1f}%",
        "Percentage Active Members": f"{(churned_customers['IsActiveMember'].mean() * 100):.1f}%",
        "Most Common Geography": churned_customers['Geography'].mode().iloc[0],
        "Gender Distribution": f"Female: {(churned_customers['Gender'] == 'Female').mean() * 100:.1f}%, Male: {(churned_customers['Gender'] == 'Male').mean() * 100:.1f}%"
    }

def get_ai_explanation(prompt: str) -> str:
    """
    Get explanation from Groq AI using the generated prompt.
    
    Args:
        prompt: The formatted prompt for AI explanation
        
    Returns:
        str: AI-generated explanation
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500  # Increased for more detailed responses
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# ——— UI ———
st.markdown("---")
st.markdown("### 👤 Customer Selection & Analysis")
st.markdown("Select a customer below to analyze their churn risk and adjust parameters as needed.")

selected = st.selectbox("🔍 Select a customer", customers, help="Choose a customer to analyze their churn probability")

if selected:
    cust_id = int(selected.split(" - ")[0])
    cust_row = df.loc[df["CustomerId"] == cust_id].iloc[0]

    st.markdown("### ⚙️ Customer Profile & Parameters")
    st.markdown("Adjust the customer parameters below to see how they affect the churn prediction.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Financial Information**")
        credit_score     = st.number_input("💳 Credit Score",      300, 850, int(cust_row["CreditScore"]), help="Customer's credit score (300-850)")
        balance          = st.number_input("💰 Balance",             0.0, 250000.0, float(cust_row["Balance"]), help="Current account balance")
        estimated_salary = st.number_input("💵 Estimated Salary",    0.0, 200000.0, float(cust_row["EstimatedSalary"]), help="Estimated annual salary")
        
        st.markdown("**📈 Banking Relationship**")
        tenure           = st.number_input("⏱️ Tenure (years)",      0,  50, int(cust_row["Tenure"]), help="Years as a customer")
        num_of_products   = st.number_input("🛍️ Number of Products",  1,    4, int(cust_row["NumOfProducts"]), help="Number of bank products")

    with col2:
        st.markdown("**👤 Personal Information**")
        location          = st.selectbox("🌍 Geography", locations,
                                         index=locations.index(cust_row["Geography"]), help="Customer's location")
        age               = st.number_input("🎂 Age",                18,  100, int(cust_row["Age"]), help="Customer's age")
        gender           = st.radio("👤 Gender", ["Male", "Female"],
                                    index=0 if cust_row["Gender"]=="Male" else 1, help="Customer's gender")
        
        st.markdown("**🏦 Account Features**")
        has_credit_card   = st.radio("💳 Has Credit Card", ["Yes","No"],
                                     index=0 if cust_row["HasCrCard"]==1 else 1, help="Does the customer have a credit card?")
        has_active_member = st.radio("✅ Active Member", ["Yes","No"],
                                     index=0 if cust_row["IsActiveMember"]==1 else 1, help="Is the customer an active member?")

    # Prepare inputs
    input_df, input_dict = prepare_input(
        credit_score,
        location,
        gender,
        age,
        tenure,
        balance,
        num_of_products,
        has_credit_card == "Yes",
        has_active_member == "Yes",
        estimated_salary,
        locations,
    )
    # Make and display predictions
    probs = make_predictions(input_df)
    display_predictions(probs)
    
    # Add explanation section with better UI
    st.markdown("---")
    
    # Create columns for the explanation section
    exp_col1, exp_col2 = st.columns([2, 1])
    
    with exp_col1:
        st.markdown("### 🤖 AI-Powered Analysis")
        st.markdown("Get detailed insights about this customer's churn risk and actionable recommendations.")
    
    with exp_col2:
        if st.button("🔍 Generate AI Explanation", type="primary", use_container_width=True):
            if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
                st.error("⚠️ Groq API key not configured. Please set your GROQ_API_KEY in the .env file.")
            else:
                with st.spinner("🧠 AI is analyzing the customer profile..."):
                    # Get average probability
                    avg_prob = np.mean(list(probs.values()))
                    
                    # Get feature importance and churned customer stats
                    feature_importance = get_feature_importance()
                    churned_stats = get_churned_customer_stats()
                    
                    # Generate the prompt
                    prompt = explain_prediction(input_dict, avg_prob, feature_importance, churned_stats)
                    
                    # Get AI explanation
                    explanation = get_ai_explanation(prompt)
                    
                    # Display the explanation in a nice container
                    st.markdown("### 📊 Detailed Analysis")
                    
                    # Format the explanation for HTML display
                    formatted_explanation = explanation.replace('\n', '<br>')
                    
                    # Create an attractive container for the explanation
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 25px;
                        border-radius: 15px;
                        color: white;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                        margin: 20px 0;
                    '>
                        <h3 style='color: white; margin-top: 0; display: flex; align-items: center;'>
                            🎯 Expert AI Analysis
                        </h3>
                        <div style='
                            background: rgba(255,255,255,0.1);
                            padding: 20px;
                            border-radius: 10px;
                            backdrop-filter: blur(10px);
                            line-height: 1.6;
                            font-size: 16px;
                        '>
                            {formatted_explanation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add key insights summary
                    avg_prob_pct = avg_prob * 100
                    if avg_prob_pct < 25:
                        insight_color = "#28a745"
                        insight_icon = "✅"
                        insight_text = "Low churn risk - Customer is likely to stay"
                    elif avg_prob_pct < 50:
                        insight_color = "#ffc107"
                        insight_icon = "⚠️"
                        insight_text = "Moderate churn risk - Monitor closely"
                    else:
                        insight_color = "#dc3545"
                        insight_icon = "🚨"
                        insight_text = "High churn risk - Immediate action needed"
                    
                    st.markdown(f"""
                    <div style='
                        background-color: {insight_color};
                        color: white;
                        padding: 15px;
                        border-radius: 10px;
                        text-align: center;
                        font-weight: bold;
                        font-size: 18px;
                        margin: 20px 0;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    '>
                        {insight_icon} {insight_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Optionally show the prompt used (for debugging)
                    with st.expander("🔧 View Generated Prompt (Advanced)"):
                        st.code(prompt, language="text")