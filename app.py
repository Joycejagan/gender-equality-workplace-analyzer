"""
AI-Based Gender Equality and Workplace Opportunity Gap Analyzer
IBM SkillsBuild Final Project
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import os

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gender Equality & Workplace Gap Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fb; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #3b82f6;
        margin-bottom: 12px;
    }
    .metric-card h3 { margin: 0; color: #374151; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card h1 { margin: 4px 0 0 0; color: #1e3a5f; font-size: 2rem; font-weight: 700; }
    .metric-card.purple { border-left-color: #8b5cf6; }
    .metric-card.green  { border-left-color: #10b981; }
    .metric-card.orange { border-left-color: #f59e0b; }
    .metric-card.red    { border-left-color: #ef4444; }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2563eb 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px 0 16px 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Info banners */
    .info-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 14px 18px;
        color: #1e40af;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    .warning-banner {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 14px 18px;
        color: #92400e;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg, section[data-testid="stSidebar"] { background-color: #1e3a5f !important; }
    
    /* Tables */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .styled-table th { background: #1e3a5f; color: white; padding: 10px 14px; text-align: left; }
    .styled-table td { padding: 9px 14px; border-bottom: 1px solid #e5e7eb; }
    .styled-table tr:hover td { background: #f0f7ff; }
    
    /* Model result box */
    .result-box {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
    .result-likely   { background: #fef2f2; border: 2px solid #fca5a5; color: #b91c1c; }
    .result-unlikely { background: #f0fdf4; border: 2px solid #86efac; color: #166534; }
    
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data loading & caching
# ─────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-HR-Employee-Attrition.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

@st.cache_resource
def train_models(df):
    """Preprocess data, train both models, return artefacts."""
    data = df.copy()

    # Drop irrelevant columns
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    data.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Encode target
    data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})

    # Identify categorical columns
    cat_cols = data.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Label-encode categoricals
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
            "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        }

    lr_metrics = metrics(y_test, y_pred_lr)
    rf_metrics = metrics(y_test, y_pred_rf)

    # Select best model (by F1)
    best_model_name = "Random Forest" if rf_metrics["F1-Score"] >= lr_metrics["F1-Score"] else "Logistic Regression"

    return {
        "lr": lr, "rf": rf, "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
        "lr_metrics": lr_metrics, "rf_metrics": rf_metrics,
        "best_model_name": best_model_name,
        "y_test": y_test,
        "y_pred_lr": y_pred_lr, "y_pred_rf": y_pred_rf,
        "X_test": X_test,
    }


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def metric_card(title, value, color="blue"):
    colors = {"blue":"", "purple":" purple", "green":" green", "orange":" orange", "red":" red"}
    cls = colors.get(color, "")
    st.markdown(f"""
    <div class="metric-card{cls}">
        <h3>{title}</h3>
        <h1>{value}</h1>
    </div>""", unsafe_allow_html=True)

def section_header(title):
    st.markdown(f'<div class="section-header">📊 {title}</div>', unsafe_allow_html=True)

def info_box(text):
    st.markdown(f'<div class="info-banner">ℹ️ {text}</div>', unsafe_allow_html=True)

def warning_box(text):
    st.markdown(f'<div class="warning-banner">⚠️ {text}</div>', unsafe_allow_html=True)

def fmt(val):
    """Format float to 2 dp string."""
    return f"{val:.2f}"


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = load_data()
artefacts = train_models(df)


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Gender Equality\n### Workplace Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home",
         "📂 Dataset Overview",
         "⚖️ Gender Equality",
         "🤖 ML Prediction",
         "📈 Model Performance",
         "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** IBM HR Analytics")
    st.markdown(f"**Employees:** {len(df):,}")
    st.markdown(f"**Best Model:** {artefacts['best_model_name']}")
    st.markdown(f"**Model Accuracy:** {artefacts['rf_metrics']['Accuracy'] if artefacts['best_model_name']=='Random Forest' else artefacts['lr_metrics']['Accuracy']}%")


# ═══════════════════════════════════════════════
# PAGE 1 – HOME
# ═══════════════════════════════════════════════
if page == "🏠 Home":
    st.title("⚖️ AI-Based Gender Equality & Workplace Opportunity Gap Analyzer")
    st.markdown("#### IBM SkillsBuild Final Project — IBM HR Analytics Employee Attrition Dataset")
    st.markdown("---")

    warning_box(
        "This system identifies gender-based differences and patterns in workplace outcomes "
        "that <b>may require further investigation</b>. Differences found in the dataset do "
        "<b>NOT</b> automatically prove discrimination. The purpose is to support data-driven "
        "analysis of gender equality, not to make unfair assumptions about individual employees."
    )

    st.markdown("### What this application does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📂 Dataset Overview**
        - Explore the IBM HR dataset
        - View employee distributions
        - Understand key statistics
        """)
    with col2:
        st.markdown("""
        **⚖️ Gender Equality Analysis**
        - Compare pay, level & promotion
        - Attrition & training gaps
        - Department representation
        """)
    with col3:
        st.markdown("""
        **🤖 ML Prediction**
        - Predict employee attrition
        - Logistic Regression vs Random Forest
        - Interactive prediction form
        """)

    st.markdown("---")

    # Quick KPI row
    total   = len(df)
    female  = len(df[df["Gender"] == "Female"])
    male    = len(df[df["Gender"] == "Male"])
    att_pct = round(len(df[df["Attrition"] == "Yes"]) / total * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Employees", f"{total:,}", "blue")
    with c2: metric_card("Female Employees", f"{female:,}", "purple")
    with c3: metric_card("Male Employees", f"{male:,}", "green")
    with c4: metric_card("Overall Attrition Rate", f"{att_pct}%", "red")

    st.markdown("---")
    st.markdown("#### Quick Gender Snapshot")

    gender_inc = df.groupby("Gender")["MonthlyIncome"].mean().reset_index()
    gender_att = df.groupby("Gender")["Attrition"].apply(lambda x: round((x=="Yes").sum()/len(x)*100,1)).reset_index()
    gender_att.columns = ["Gender", "Attrition Rate (%)"]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(gender_inc, x="Gender", y="MonthlyIncome",
                     color="Gender",
                     color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                     title="Average Monthly Income by Gender",
                     labels={"MonthlyIncome":"Avg Monthly Income ($)"},
                     text_auto=".0f")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(gender_att, x="Gender", y="Attrition Rate (%)",
                      color="Gender",
                      color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                      title="Attrition Rate by Gender (%)",
                      text_auto=".1f")
        fig2.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2 – DATASET OVERVIEW
# ═══════════════════════════════════════════════
elif page == "📂 Dataset Overview":
    st.title("📂 Dataset Overview")
    st.markdown("Exploring the IBM HR Analytics Employee Attrition Dataset")
    st.markdown("---")

    total   = len(df)
    female  = len(df[df["Gender"] == "Female"])
    male    = len(df[df["Gender"] == "Male"])
    depts   = df["Department"].nunique()
    roles   = df["JobRole"].nunique()
    att_cnt = len(df[df["Attrition"] == "Yes"])
    att_pct = round(att_cnt / total * 100, 1)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Total Employees", f"{total:,}", "blue")
    with c2: metric_card("Female Employees", f"{female:,}", "purple")
    with c3: metric_card("Male Employees", f"{male:,}", "green")
    c4, c5, c6 = st.columns(3)
    with c4: metric_card("Departments", f"{depts}", "orange")
    with c5: metric_card("Job Roles", f"{roles}", "blue")
    with c6: metric_card("Attrition Count", f"{att_cnt} ({att_pct}%)", "red")

    st.markdown("---")

    # Department & Role distributions
    section_header("Department & Job Role Distribution")
    c1, c2 = st.columns(2)
    with c1:
        dept_counts = df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig = px.pie(dept_counts, names="Department", values="Count",
                     title="Employees by Department",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        role_counts = df["JobRole"].value_counts().reset_index()
        role_counts.columns = ["JobRole", "Count"]
        fig2 = px.bar(role_counts, x="Count", y="JobRole", orientation="h",
                      title="Employees by Job Role",
                      color="Count",
                      color_continuous_scale="Blues")
        fig2.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Attrition breakdown
    section_header("Attrition Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        att_cnt_df = df["Attrition"].value_counts().reset_index()
        att_cnt_df.columns = ["Attrition", "Count"]
        fig = px.pie(att_cnt_df, names="Attrition", values="Count",
                     title="Overall Attrition Distribution",
                     color_discrete_map={"Yes":"#ef4444","No":"#22c55e"},
                     hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        dept_att = df.groupby("Department")["Attrition"].apply(
            lambda x: round((x=="Yes").sum()/len(x)*100,1)).reset_index()
        dept_att.columns = ["Department","Attrition Rate (%)"]
        fig2 = px.bar(dept_att, x="Department", y="Attrition Rate (%)",
                      title="Attrition Rate by Department (%)",
                      color="Department",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      text_auto=".1f")
        fig2.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Age distribution
    section_header("Age & Income Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age", color="Gender",
                           barmode="overlay",
                           color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                           title="Age Distribution by Gender",
                           opacity=0.75)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(df, x="Department", y="MonthlyIncome",
                      color="Department",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Monthly Income Distribution by Department")
        fig2.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    section_header("Dataset Sample (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)

    section_header("Dataset Column Information")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Sample Value": [str(df[c].iloc[0]) for c in df.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 3 – GENDER EQUALITY
# ═══════════════════════════════════════════════
elif page == "⚖️ Gender Equality":
    st.title("⚖️ Gender Equality Analysis")
    st.markdown("Comparing workplace outcomes between Male and Female employees using real dataset values.")

    warning_box(
        "The differences identified below are based on the dataset and represent observable patterns. "
        "They do <b>NOT</b> automatically prove discrimination or bias. "
        "They are indicators that may require further investigation by HR professionals."
    )
    st.markdown("---")

    male_df   = df[df["Gender"] == "Male"]
    female_df = df[df["Gender"] == "Female"]

    # Summary comparison table
    section_header("Key Metrics Comparison: Male vs Female")

    metrics_table = {
        "Metric": [
            "Total Count",
            "Avg Monthly Income ($)",
            "Avg Job Level",
            "Avg Training Sessions / Year",
            "Avg Years Since Last Promotion",
            "Avg Years at Company",
            "Avg Performance Rating",
            "Avg Job Satisfaction",
            "Attrition Rate (%)",
        ],
        "Male": [
            f"{len(male_df):,}",
            f"{male_df['MonthlyIncome'].mean():.0f}",
            f"{male_df['JobLevel'].mean():.2f}",
            f"{male_df['TrainingTimesLastYear'].mean():.2f}",
            f"{male_df['YearsSinceLastPromotion'].mean():.2f}",
            f"{male_df['YearsAtCompany'].mean():.2f}",
            f"{male_df['PerformanceRating'].mean():.2f}",
            f"{male_df['JobSatisfaction'].mean():.2f}",
            f"{(male_df['Attrition']=='Yes').sum()/len(male_df)*100:.1f}%",
        ],
        "Female": [
            f"{len(female_df):,}",
            f"{female_df['MonthlyIncome'].mean():.0f}",
            f"{female_df['JobLevel'].mean():.2f}",
            f"{female_df['TrainingTimesLastYear'].mean():.2f}",
            f"{female_df['YearsSinceLastPromotion'].mean():.2f}",
            f"{female_df['YearsAtCompany'].mean():.2f}",
            f"{female_df['PerformanceRating'].mean():.2f}",
            f"{female_df['JobSatisfaction'].mean():.2f}",
            f"{(female_df['Attrition']=='Yes').sum()/len(female_df)*100:.1f}%",
        ]
    }
    cmp_df = pd.DataFrame(metrics_table)
    st.markdown(cmp_df.to_html(index=False, classes="styled-table"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts row 1: Income & Job Level ---
    section_header("Income & Career Level Comparison")
    c1, c2 = st.columns(2)
    with c1:
        inc_data = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Avg Monthly Income ($)": [
                round(male_df["MonthlyIncome"].mean(), 0),
                round(female_df["MonthlyIncome"].mean(), 0)
            ]
        })
        fig = px.bar(inc_data, x="Gender", y="Avg Monthly Income ($)",
                     color="Gender",
                     color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                     title="Average Monthly Income",
                     text_auto=".0f")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        lvl_data = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Avg Job Level": [
                round(male_df["JobLevel"].mean(), 2),
                round(female_df["JobLevel"].mean(), 2)
            ]
        })
        fig2 = px.bar(lvl_data, x="Gender", y="Avg Job Level",
                      color="Gender",
                      color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                      title="Average Job Level",
                      text_auto=".2f")
        fig2.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Income by job level and gender
    inc_lvl = df.groupby(["JobLevel","Gender"])["MonthlyIncome"].mean().reset_index()
    inc_lvl.columns = ["JobLevel","Gender","Avg Income"]
    fig = px.bar(inc_lvl, x="JobLevel", y="Avg Income", color="Gender",
                 barmode="group",
                 color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                 title="Average Monthly Income by Job Level and Gender",
                 labels={"JobLevel":"Job Level","Avg Income":"Avg Monthly Income ($)"})
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

    # --- Charts row 2: Attrition & Training ---
    section_header("Attrition & Training Comparison")
    c1, c2 = st.columns(2)
    with c1:
        att_data = pd.DataFrame({
            "Gender": ["Male","Female"],
            "Attrition Rate (%)": [
                round((male_df["Attrition"]=="Yes").sum()/len(male_df)*100, 1),
                round((female_df["Attrition"]=="Yes").sum()/len(female_df)*100, 1)
            ]
        })
        fig = px.bar(att_data, x="Gender", y="Attrition Rate (%)",
                     color="Gender",
                     color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                     title="Attrition Rate by Gender (%)",
                     text_auto=".1f")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tr_data = pd.DataFrame({
            "Gender": ["Male","Female"],
            "Avg Training Sessions": [
                round(male_df["TrainingTimesLastYear"].mean(), 2),
                round(female_df["TrainingTimesLastYear"].mean(), 2)
            ]
        })
        fig2 = px.bar(tr_data, x="Gender", y="Avg Training Sessions",
                      color="Gender",
                      color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                      title="Average Training Sessions Last Year",
                      text_auto=".2f")
        fig2.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Charts row 3: Promotion & Satisfaction ---
    section_header("Promotion & Satisfaction Comparison")
    c1, c2 = st.columns(2)
    with c1:
        promo_data = pd.DataFrame({
            "Gender": ["Male","Female"],
            "Avg Years Since Last Promotion": [
                round(male_df["YearsSinceLastPromotion"].mean(), 2),
                round(female_df["YearsSinceLastPromotion"].mean(), 2)
            ]
        })
        fig = px.bar(promo_data, x="Gender", y="Avg Years Since Last Promotion",
                     color="Gender",
                     color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                     title="Average Years Since Last Promotion",
                     text_auto=".2f")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sat_data = pd.DataFrame({
            "Gender": ["Male","Female"],
            "Avg Job Satisfaction": [
                round(male_df["JobSatisfaction"].mean(), 2),
                round(female_df["JobSatisfaction"].mean(), 2)
            ]
        })
        fig2 = px.bar(sat_data, x="Gender", y="Avg Job Satisfaction",
                      color="Gender",
                      color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                      title="Average Job Satisfaction Score",
                      text_auto=".2f")
        fig2.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Department & Role Representation ---
    section_header("Gender Representation Across Departments and Job Roles")
    dept_gender = df.groupby(["Department","Gender"]).size().reset_index(name="Count")
    fig = px.bar(dept_gender, x="Department", y="Count", color="Gender",
                 barmode="group",
                 color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                 title="Employee Count by Department and Gender")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    role_gender = df.groupby(["JobRole","Gender"]).size().reset_index(name="Count")
    role_total  = df.groupby("JobRole").size().reset_index(name="Total")
    role_gender = role_gender.merge(role_total, on="JobRole")
    role_gender["Pct"] = round(role_gender["Count"] / role_gender["Total"] * 100, 1)
    fig2 = px.bar(role_gender, x="JobRole", y="Pct", color="Gender",
                  barmode="stack",
                  color_discrete_map={"Female":"#8b5cf6","Male":"#3b82f6"},
                  title="Gender Distribution (%) by Job Role",
                  labels={"Pct":"Percentage (%)"},
                  text_auto=".0f")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Overtime
    section_header("Overtime Distribution by Gender")
    ot_gender = df.groupby(["Gender","OverTime"]).size().reset_index(name="Count")
    fig = px.bar(ot_gender, x="Gender", y="Count", color="OverTime",
                 barmode="group",
                 color_discrete_map={"Yes":"#ef4444","No":"#22c55e"},
                 title="Overtime Status by Gender")
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

    info_box(
        "Note: All values above are calculated directly from the IBM HR Analytics dataset. "
        "Observed differences should be interpreted in the context of the organisation's structure, "
        "hiring history and roles — they do not automatically indicate gender discrimination."
    )


# ═══════════════════════════════════════════════
# PAGE 4 – ML PREDICTION
# ═══════════════════════════════════════════════
elif page == "🤖 ML Prediction":
    st.title("🤖 Attrition Prediction")
    st.markdown("Enter employee information below to predict whether they are likely to leave the company.")

    info_box(
        f"The best-performing model (<b>{artefacts['best_model_name']}</b>) is used for prediction. "
        "This model was trained on 80% of the IBM HR dataset and evaluated on the remaining 20%."
    )
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("### 👤 Employee Information")

        c1, c2, c3 = st.columns(3)
        with c1:
            age          = st.number_input("Age", 18, 65, 35)
            gender       = st.selectbox("Gender", ["Female", "Male"])
            marital      = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education    = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
            edu_field    = st.selectbox("Education Field",
                                        ["Life Sciences","Medical","Marketing",
                                         "Technical Degree","Other","Human Resources"])
        with c2:
            department   = st.selectbox("Department",
                                        ["Sales","Research & Development","Human Resources"])
            job_role     = st.selectbox("Job Role",
                                        ["Sales Executive","Research Scientist",
                                         "Laboratory Technician","Manufacturing Director",
                                         "Healthcare Representative","Manager",
                                         "Sales Representative","Research Director",
                                         "Human Resources"])
            job_level    = st.slider("Job Level (1–5)", 1, 5, 2)
            job_satisf   = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
            job_involv   = st.slider("Job Involvement (1–4)", 1, 4, 3)
        with c3:
            monthly_inc  = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
            overtime     = st.selectbox("OverTime", ["No","Yes"])
            business_tr  = st.selectbox("Business Travel",
                                        ["Non-Travel","Travel_Rarely","Travel_Frequently"])
            wlb          = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
            perf_rating  = st.slider("Performance Rating (1–4)", 1, 4, 3)

        st.markdown("### 📅 Experience & Career")
        c4, c5, c6 = st.columns(3)
        with c4:
            years_company  = st.number_input("Years at Company", 0, 40, 5)
            years_role     = st.number_input("Years in Current Role", 0, 20, 3)
            years_promo    = st.number_input("Years Since Last Promotion", 0, 20, 2)
            years_manager  = st.number_input("Years with Current Manager", 0, 20, 3)
        with c5:
            training       = st.slider("Training Times Last Year", 0, 6, 3)
            total_working  = st.number_input("Total Working Years", 0, 40, 10)
            num_companies  = st.number_input("Num Companies Worked", 0, 10, 2)
            stock_opt      = st.slider("Stock Option Level (0–3)", 0, 3, 1)
        with c6:
            distance_home  = st.number_input("Distance From Home (km)", 1, 100, 10)
            daily_rate     = st.number_input("Daily Rate ($)", 100, 1500, 800)
            hourly_rate    = st.number_input("Hourly Rate ($)", 30, 100, 65)
            monthly_rate   = st.number_input("Monthly Rate ($)", 2000, 27000, 14000)
            pct_hike       = st.slider("Percent Salary Hike", 11, 25, 14)
            env_satisf     = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
            rel_satisf     = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)

        submitted = st.form_submit_button("🔮 Predict Attrition", use_container_width=True)

    if submitted:
        # Build input dict using the exact column order from training
        le = artefacts["label_encoders"]
        feature_names = artefacts["feature_names"]

        def safe_encode(enc_dict, col, val):
            encoder = enc_dict.get(col)
            if encoder is None:
                return val
            classes = list(encoder.classes_)
            if val in classes:
                return encoder.transform([val])[0]
            return 0  # fallback for unseen labels

        input_map = {
            "Age":                     age,
            "BusinessTravel":          safe_encode(le, "BusinessTravel", business_tr),
            "DailyRate":               daily_rate,
            "Department":              safe_encode(le, "Department", department),
            "DistanceFromHome":        distance_home,
            "Education":               education,
            "EducationField":          safe_encode(le, "EducationField", edu_field),
            "EnvironmentSatisfaction": env_satisf,
            "Gender":                  safe_encode(le, "Gender", gender),
            "HourlyRate":              hourly_rate,
            "JobInvolvement":          job_involv,
            "JobLevel":                job_level,
            "JobRole":                 safe_encode(le, "JobRole", job_role),
            "JobSatisfaction":         job_satisf,
            "MaritalStatus":           safe_encode(le, "MaritalStatus", marital),
            "MonthlyIncome":           monthly_inc,
            "MonthlyRate":             monthly_rate,
            "NumCompaniesWorked":      num_companies,
            "OverTime":                safe_encode(le, "OverTime", overtime),
            "PercentSalaryHike":       pct_hike,
            "PerformanceRating":       perf_rating,
            "RelationshipSatisfaction":rel_satisf,
            "StockOptionLevel":        stock_opt,
            "TotalWorkingYears":       total_working,
            "TrainingTimesLastYear":   training,
            "WorkLifeBalance":         wlb,
            "YearsAtCompany":          years_company,
            "YearsInCurrentRole":      years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager":    years_manager,
        }

        # Build input array in the exact feature order
        input_values = [input_map.get(f, 0) for f in feature_names]
        X_input = np.array([input_values])

        # Predict with best model
        if artefacts["best_model_name"] == "Random Forest":
            model = artefacts["rf"]
            pred  = model.predict(X_input)[0]
            prob  = model.predict_proba(X_input)[0]
        else:
            model  = artefacts["lr"]
            scaler = artefacts["scaler"]
            X_sc   = scaler.transform(X_input)
            pred   = model.predict(X_sc)[0]
            prob   = model.predict_proba(X_sc)[0]

        prob_leave = round(prob[1] * 100, 1)
        prob_stay  = round(prob[0] * 100, 1)

        st.markdown("---")
        st.markdown("### 🔮 Prediction Result")

        if pred == 1:
            st.markdown(f"""
            <div class="result-box result-likely">
                <h2>⚠️ Predicted Attrition: <b>LIKELY</b></h2>
                <p style="font-size:1.2rem">This employee is <b>likely to leave</b> based on the model.</p>
                <p style="font-size:1rem">Probability of leaving: <b>{prob_leave}%</b> &nbsp;|&nbsp; Probability of staying: <b>{prob_stay}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-unlikely">
                <h2>✅ Predicted Attrition: <b>UNLIKELY</b></h2>
                <p style="font-size:1.2rem">This employee is <b>likely to stay</b> based on the model.</p>
                <p style="font-size:1rem">Probability of leaving: <b>{prob_leave}%</b> &nbsp;|&nbsp; Probability of staying: <b>{prob_stay}%</b></p>
            </div>""", unsafe_allow_html=True)

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_leave,
            title={"text": "Attrition Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if pred==1 else "#22c55e"},
                "steps": [
                    {"range": [0, 30],  "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100],"color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 50},
            }
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

        warning_box(
            "This prediction is based on statistical patterns in the dataset. "
            "It should <b>NOT</b> be used as the sole basis for HR decisions. "
            "Individual circumstances always require human judgment."
        )


# ═══════════════════════════════════════════════
# PAGE 5 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Machine Learning Model Performance")
    st.markdown("Comparing Logistic Regression and Random Forest Classifier on the IBM HR dataset.")
    st.markdown("---")

    lr_m = artefacts["lr_metrics"]
    rf_m = artefacts["rf_metrics"]
    best = artefacts["best_model_name"]

    # Model comparison table
    section_header("Model Comparison")
    comp_df = pd.DataFrame({
        "Metric": ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
        "Logistic Regression": [lr_m["Accuracy"], lr_m["Precision"], lr_m["Recall"], lr_m["F1-Score"]],
        "Random Forest":       [rf_m["Accuracy"], rf_m["Precision"], rf_m["Recall"], rf_m["F1-Score"]],
    })
    st.markdown(comp_df.to_html(index=False, classes="styled-table"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.success(f"✅ **Best Model: {best}** (selected based on highest F1-Score)")

    # ── Bar chart comparison (px.bar avoids go.Figure SVG label issues) ──────
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    perf_rows = (
        [{"Metric": m, "Score": lr_m[m], "Model": "Logistic Regression"} for m in metrics_labels] +
        [{"Metric": m, "Score": rf_m[m], "Model": "Random Forest"}       for m in metrics_labels]
    )
    perf_df = pd.DataFrame(perf_rows)
    fig = px.bar(
        perf_df, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_map={"Logistic Regression": "#3b82f6", "Random Forest": "#8b5cf6"},
        title="Model Performance Comparison (%)",
        text="Score",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        height=420,
        yaxis=dict(range=[0, 115], title="Score (%)"),
        xaxis_title="Metric",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrices (go.Heatmap + manual annotations — no text_auto) ──
    section_header("Confusion Matrices")
    y_test    = artefacts["y_test"]
    y_pred_lr = artefacts["y_pred_lr"]
    y_pred_rf = artefacts["y_pred_rf"]

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    def build_cm_figure(cm, title):
        labels = ["No Attrition", "Attrition"]
        annotations = []
        for row_i, row in enumerate(cm):
            for col_j, val in enumerate(row):
                font_color = "white" if val > cm.max() / 2 else "#1e3a5f"
                annotations.append(dict(
                    x=labels[col_j], y=labels[row_i],
                    text=str(val),
                    showarrow=False,
                    font=dict(size=22, color=font_color, family="Arial Black"),
                ))
        hm = go.Heatmap(
            z=cm.tolist(),
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
            hoverongaps=False,
        )
        layout = go.Layout(
            title=dict(text=title, font=dict(size=14, color="#1e3a5f")),
            height=380,
            xaxis=dict(title="Predicted Label", side="bottom", tickfont=dict(size=12)),
            yaxis=dict(title="Actual Label", autorange="reversed", tickfont=dict(size=12)),
            annotations=annotations,
            margin=dict(l=100, r=20, t=70, b=80),
        )
        return go.Figure(data=[hm], layout=layout)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_cm_figure(cm_lr, "Confusion Matrix — Logistic Regression"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(build_cm_figure(cm_rf, "Confusion Matrix — Random Forest"),
                        use_container_width=True)

    # ── Feature Importance (solid colour bar, no color_continuous_scale) ─────
    section_header("Feature Importance (Random Forest)")
    rf_model = artefacts["rf"]
    feat_imp = pd.DataFrame({
        "Feature":    artefacts["feature_names"],
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True).tail(15)

    fig = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        title="Top 15 Most Important Features for Attrition Prediction",
        color_discrete_sequence=["#3b82f6"],
        text="Importance",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=520,
        xaxis=dict(range=[0, feat_imp["Importance"].max() * 1.25], title="Importance Score"),
        yaxis_title="",
        margin=dict(l=160, r=60, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    info_box(
        "Feature importance shows which factors the Random Forest model considers most influential "
        "in predicting employee attrition. Higher importance = more predictive power."
    )

    # Training methodology
    section_header("Training Methodology")
    st.markdown("""
    | Step | Detail |
    |------|--------|
    | **Dataset** | IBM HR Analytics Employee Attrition (1,470 employees, 35 features) |
    | **Target Variable** | Attrition (Yes=1, No=0) |
    | **Removed Columns** | EmployeeNumber, EmployeeCount, StandardHours, Over18 |
    | **Encoding** | Label Encoding for all categorical features |
    | **Scaling** | StandardScaler applied for Logistic Regression only |
    | **Train/Test Split** | 80% Training / 20% Testing (stratified) |
    | **Random State** | 42 (for reproducibility) |
    | **LR Max Iterations** | 1000 |
    | **RF Estimators** | 100 trees |
    """)


# ═══════════════════════════════════════════════
# PAGE 6 – ABOUT
# ═══════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ## 🏷️ Project Title
    **AI-Based Gender Equality and Workplace Opportunity Gap Analyzer**

    ---

    ## 📋 Problem Statement
    Gender inequality in the workplace remains a critical global issue. Many organisations lack
    the tools to identify and quantify gender-based disparities in salary, promotions, attrition,
    and training opportunities from their own data. This project addresses that gap by providing
    a data-driven analytical platform.

    ---

    ## 🎯 Objective
    Build an interactive Machine Learning-based system that:
    - Analyses IBM HR employee data to surface gender-related patterns in workplace outcomes.
    - Trains classification models to predict employee attrition.
    - Provides an interactive dashboard for non-technical HR stakeholders.
    - Supports data-driven gender equality analysis without making discriminatory assumptions.

    ---

    ## 📊 Dataset
    | Property | Value |
    |----------|-------|
    | **Name** | IBM HR Analytics Employee Attrition Dataset |
    | **File** | WA_Fn-UseC_-HR-Employee-Attrition.csv |
    | **Source** | IBM / Kaggle |
    | **Total Records** | 1,470 employees |
    | **Features** | 35 columns |
    | **Target** | Attrition (Yes / No) |

    ---

    ## 🤖 Machine Learning Algorithms Used
    | Algorithm | Type | Purpose |
    |-----------|------|---------|
    | **Logistic Regression** | Linear Classifier | Baseline model, fast and interpretable |
    | **Random Forest Classifier** | Ensemble (100 trees) | High-accuracy, handles non-linear patterns |

    ---

    ## 💻 Technologies Used
    | Technology | Purpose |
    |-----------|---------|
    | **Python 3.x** | Core programming language |
    | **Pandas** | Data manipulation and analysis |
    | **NumPy** | Numerical computing |
    | **Scikit-learn** | Machine learning models and preprocessing |
    | **Plotly** | Interactive charts and visualisations |
    | **Streamlit** | Web application framework |

    ---

    ## ⚠️ Limitations
    1. The dataset represents a fictional IBM company created for educational purposes.
    2. The ML model predicts patterns based on historical data, not future certainty.
    3. Correlation ≠ Causation: observed gender differences may have multiple explanations.
    4. Label Encoding may not capture all nuances of ordinal/nominal variables.
    5. The model may reflect historical biases present in the training data.
    6. Not suitable for real HR decision-making without expert review.

    ---

    ## 🚀 Future Scope
    1. Integrate real anonymised organisational data for genuine insights.
    2. Apply SHAP (SHapley Additive exPlanations) for explainable AI.
    3. Add time-series analysis to track gender equality trends over years.
    4. Include intersectional analysis (e.g. gender × race × age).
    5. Deploy as a cloud-based HR analytics tool (IBM Cloud / Azure).
    6. Add fairness-aware ML algorithms that actively reduce bias.
    7. Implement role-based access control for HR teams.

    ---

    ## ⚖️ Ethical Disclaimer
    """)

    warning_box(
        "This system identifies gender-based differences and patterns in workplace outcomes "
        "that <b>may require further investigation</b>. Differences found in the dataset do "
        "<b>NOT</b> automatically prove discrimination or bias. The purpose is to support "
        "data-driven analysis of gender equality, not to make unfair assumptions about "
        "individual employees. All predictions and analyses should be reviewed by qualified "
        "HR professionals before any action is taken."
    )

    st.markdown("""
    ---
    *Developed as an IBM SkillsBuild Final Project submission.*
    *Technologies: Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit*
    """)
