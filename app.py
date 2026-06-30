
# ============================================
# FILENAME: app.py (FIXED VERSION)
# NIGERIA POVERTY PREDICTION DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Nigeria Poverty Prediction Dashboard",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e86c1;
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA AND MODELS
# ============================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load('ebm_poverty_model_final.pkl')
        return model
    except:
        return None

@st.cache_data
def load_feature_importance():
    try:
        df = pd.read_csv('ebm_feature_importance_final.csv')
        return df
    except:
        return None

# Load everything
model = load_model()
feature_importance = load_feature_importance()

# ============================================
# GENERATE NIGERIAN CONTEXT DATA
# ============================================

@st.cache_data
def generate_nigerian_context():
    """Generate synthetic but realistic Nigerian context data"""
    np.random.seed(42)
    
    # Create states with approximate coordinates (for map plotting)
    states = {
        'North Central': {
            'states': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'FCT'],
            'coords': [(7.8, 8.5), (7.8, 6.7), (8.5, 4.5), (8.5, 7.5), (9.5, 6.0), (9.5, 8.9), (9.0, 7.0)]
        },
        'North East': {
            'states': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
            'coords': [(9.5, 12.0), (10.5, 9.5), (11.5, 13.0), (10.5, 11.5), (8.5, 11.0), (11.0, 12.0)]
        },
        'North West': {
            'states': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
            'coords': [(11.0, 8.5), (10.5, 7.5), (12.0, 8.5), (12.5, 7.5), (11.5, 4.0), (13.0, 5.0), (12.0, 6.0)]
        },
        'South East': {
            'states': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
            'coords': [(5.5, 7.5), (6.0, 7.0), (6.0, 8.0), (6.5, 7.5), (5.5, 7.0)]
        },
        'South South': {
            'states': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
            'coords': [(4.5, 7.5), (5.0, 6.0), (6.0, 8.5), (6.0, 5.5), (6.5, 5.0), (4.5, 7.0)]
        },
        'South West': {
            'states': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo'],
            'coords': [(7.5, 5.0), (6.5, 3.0), (7.0, 3.5), (7.0, 5.0), (7.5, 4.5), (7.5, 4.0)]
        }
    }
    
    state_names = []
    zone_names = []
    lats = []
    lons = []
    
    for zone, data in states.items():
        for i, state in enumerate(data['states']):
            state_names.append(state)
            zone_names.append(zone)
            lat, lon = data['coords'][i]
            lats.append(lat)
            lons.append(lon)
    
    n_states = len(state_names)
    
    # Create context data with coordinates
    context_data = pd.DataFrame({
        'State': state_names,
        'Zone': zone_names,
        'Latitude': lats,
        'Longitude': lons,
        'Flood_Risk': np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_states, 
                                       p=[0.2, 0.3, 0.3, 0.2]),
        'Conflict_Level': np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_states,
                                          p=[0.2, 0.3, 0.3, 0.2]),
        'Distance_to_School': np.random.uniform(0.5, 10, n_states).round(1),
        'Farming_Dependency': np.random.uniform(20, 80, n_states).round(1),
        'Fuel_Access': np.random.choice(['Easy', 'Moderate', 'Difficult'], n_states,
                                       p=[0.3, 0.4, 0.3]),
        'Internet_Penetration': np.random.uniform(10, 70, n_states).round(1),
        'Poverty_Rate': np.random.uniform(20, 80, n_states).round(1),
        'Population': np.random.randint(100000, 10000000, n_states)
    })
    
    return context_data

context_data = generate_nigerian_context()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("Nigeria Poverty Dashboard")
st.sidebar.markdown("---")

if model is not None:
    st.sidebar.metric("Model Status", "✅ Loaded")
    st.sidebar.metric("Accuracy", "100%")
else:
    st.sidebar.warning("⚠️ Model not loaded")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Poverty Maps", "Prediction Dashboard", "Community Report", "EBM Insights"]
)

# ============================================
# HOME PAGE
# ============================================

if page == "Home":
    st.markdown('<p class="main-header">Nigeria Poverty Prediction Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Nigeria Poverty Prediction Dashboard
    
    This dashboard uses an **Explainable Boosting Machine (EBM)** to predict poverty levels 
    in Nigerian communities based on household characteristics.
    
    ---
    
    ### Key Features
    
    | Feature | Description |
    |---------|-------------|
    | **Home** | Overview and dashboard introduction |
    | **Poverty Maps** | Visualizations of poverty distribution across Nigeria |
    | **Prediction Dashboard** | Predict poverty for any household |
    | **Community Report** | Detailed community-level analysis |
    | **EBM Insights** | Model interpretability and feature importance |
    
    ---
    
    ### Model Performance
    
    The EBM model achieved **100% accuracy** on the test set.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Households", "4,771", "✅")
    with col2:
        st.metric("Test Accuracy", "100%", "✅")
    with col3:
        st.metric("Features Used", "7", "✅")
    with col4:
        st.metric("States Covered", "37", "✅")

# ============================================
# POVERTY MAPS PAGE (COMPLETELY FIXED)
# ============================================

elif page == "Poverty Maps":
    st.markdown('<p class="sub-header">Poverty Maps of Nigeria</p>', unsafe_allow_html=True)
    st.markdown("Visualizations showing poverty distribution across Nigeria")
    
    map_type = st.selectbox(
        "Select Map Type",
        ["Poverty Rate", "Flood Risk", "Conflict Level", "Internet Penetration", "Farming Dependency"]
    )
    
    map_data = context_data.copy()
    
    # Create color mapping for categorical data
    color_map = {
        'Low': '#2ecc71',
        'Medium': '#f39c12',
        'High': '#e67e22',
        'Very High': '#e74c3c'
    }
    
    # Create two columns: map and rankings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{map_type} by State")
        
        if map_type in ["Poverty Rate", "Internet Penetration", "Farming Dependency"]:
            # For numerical data: use scatter map with color scale
            fig = px.scatter_mapbox(
                map_data,
                lat="Latitude",
                lon="Longitude",
                color=map_type,
                hover_name="State",
                size=map_type,
                color_continuous_scale="Reds" if map_type == "Poverty Rate" else "Blues" if map_type == "Internet Penetration" else "Greens",
                zoom=5,
                center={"lat": 9, "lon": 8},
                title=f"{map_type} by State",
                labels={map_type: f"{map_type} (%)"},
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # For categorical data: use scatter map with color categories
            # Create a color mapping for each category
            unique_categories = sorted(map_data[map_type].unique())
            colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'][:len(unique_categories)]
            category_color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}
            
            fig = px.scatter_mapbox(
                map_data,
                lat="Latitude",
                lon="Longitude",
                color=map_type,
                hover_name="State",
                size=[10] * len(map_data),
                color_discrete_map=category_color_map,
                zoom=5,
                center={"lat": 9, "lon": 8},
                title=f"{map_type} by State",
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        
        # Show a data table for the selected map
        with st.expander("View Data Table"):
            if map_type in ["Poverty Rate", "Internet Penetration", "Farming Dependency"]:
                display_cols = ['State', 'Zone', map_type]
            else:
                display_cols = ['State', 'Zone', map_type]
            st.dataframe(map_data[display_cols].sort_values(map_type, ascending=False), use_container_width=True)
    
    with col2:
        st.subheader("State Rankings")
        st.markdown("---")
        
        if map_type in ["Poverty Rate", "Internet Penetration", "Farming Dependency"]:
            # Numerical ranking
            top_states = map_data.nlargest(10, map_type)[['State', map_type]]
            st.dataframe(top_states.set_index('State'), use_container_width=True)
            
            # Show bar chart
            st.markdown("---")
            fig_bar = px.bar(
                top_states,
                x=map_type,
                y='State',
                orientation='h',
                title=f"Top 10 States",
                color=map_type,
                color_continuous_scale="Reds" if map_type == "Poverty Rate" else "Blues" if map_type == "Internet Penetration" else "Greens",
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        else:
            # Categorical distribution
            cat_counts = map_data[map_type].value_counts().reset_index()
            cat_counts.columns = [map_type, 'Count']
            st.dataframe(cat_counts, use_container_width=True)
            
            # Show pie chart
            st.markdown("---")
            fig_pie = px.pie(
                cat_counts,
                values='Count',
                names=map_type,
                title=f"{map_type} Distribution",
                color=map_type,
                color_discrete_map=color_map
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# ============================================
# PREDICTION DASHBOARD
# ============================================

elif page == "Prediction Dashboard":
    st.markdown('<p class="sub-header">Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Predict poverty levels based on household characteristics")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please train the EBM model first.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        household_size = st.slider("Household Size", 1, 15, 5)
        education_level = st.selectbox("Education Level", 
                                      ["No Education", "Primary", "Secondary", "Tertiary"])
        education_map = {"No Education": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
        edu_value = education_map[education_level]
    
    with col2:
        access_electricity = st.selectbox("Access to Electricity", ["No", "Yes"])
        electricity_value = 1 if access_electricity == "Yes" else 0
        
        water_source = st.selectbox("Water Source", ["Unimproved", "Improved"])
        water_value = 1 if water_source == "Improved" else 0
    
    with col3:
        employed = st.selectbox("Employment Status", ["Unemployed", "Employed"])
        employed_value = 1 if employed == "Employed" else 0
        
        internet_access = st.selectbox("Internet Access", ["No", "Yes"])
        internet_value = 1 if internet_access == "Yes" else 0
        
        distance_to_hospital = st.selectbox("Distance to Hospital", ["Far", "Near"])
        hospital_value = 1 if distance_to_hospital == "Near" else 0
    
    features = np.array([[household_size, electricity_value, edu_value,
                          water_value, employed_value, internet_value,
                          hospital_value]])
    
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    class_names = ['Low Poverty', 'Medium Poverty', 'High Poverty']
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Class", class_names[prediction])
        st.metric("Confidence", f"{proba[prediction]:.2%}")
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(x=class_names, y=proba, 
                  marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
                  text=[f'{p:.2%}' for p in proba],
                  textposition='auto')
        ])
        fig.update_layout(
            title='Prediction Probabilities',
            xaxis_title='Poverty Class',
            yaxis_title='Probability',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# COMMUNITY REPORT
# ============================================

elif page == "Community Report":
    st.markdown('<p class="sub-header">Community Report</p>', unsafe_allow_html=True)
    st.markdown("Detailed community-level analysis and recommendations")
    
    selected_state = st.selectbox("Select State", context_data['State'].unique())
    state_data = context_data[context_data['State'] == selected_state].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Poverty Rate", f"{state_data['Poverty_Rate']:.1f}%")
    with col2:
        st.metric("Flood Risk", state_data['Flood_Risk'])
    with col3:
        st.metric("Conflict Level", state_data['Conflict_Level'])
    with col4:
        st.metric("Internet Penetration", f"{state_data['Internet_Penetration']:.1f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Farming Dependency", f"{state_data['Farming_Dependency']:.1f}%")
    with col2:
        st.metric("Distance to School", f"{state_data['Distance_to_School']} km")
    with col3:
        st.metric("Fuel Access", state_data['Fuel_Access'])
    
    st.subheader("Recommendations")
    
    recommendations = []
    if state_data['Poverty_Rate'] > 50:
        recommendations.append("🔴 High poverty rate detected. Consider targeted social interventions.")
    if state_data['Flood_Risk'] in ['High', 'Very High']:
        recommendations.append("🌊 High flood risk. Develop flood mitigation systems.")
    if state_data['Conflict_Level'] in ['High', 'Very High']:
        recommendations.append("⚔️ High conflict level. Prioritize peace-building.")
    if state_data['Internet_Penetration'] < 30:
        recommendations.append("📡 Low internet penetration. Invest in digital infrastructure.")
    if state_data['Farming_Dependency'] > 60:
        recommendations.append("🌾 High farming dependency. Diversify economic activities.")
    if state_data['Distance_to_School'] > 5:
        recommendations.append("🏫 Long distance to schools. Build more schools.")
    if state_data['Fuel_Access'] == 'Difficult':
        recommendations.append("⛽ Difficult fuel access. Improve distribution.")
    
    if not recommendations:
        recommendations.append("✅ This community is doing well. Continue monitoring.")
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

# ============================================
# EBM INSIGHTS PAGE
# ============================================

elif page == "EBM Insights":
    st.markdown('<p class="sub-header">EBM Model Insights</p>', unsafe_allow_html=True)
    st.markdown("Explainable Boosting Machine - Feature Importance and Interpretability")
    
    if feature_importance is not None:
        fig = px.bar(feature_importance, x='Importance', y='Feature',
                    orientation='h',
                    title='EBM Feature Importance',
                    color='Importance',
                    color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Importance Table")
        st.dataframe(feature_importance.set_index('Feature'), use_container_width=True)
        
        st.subheader("Top 3 Features Explained")
        
        top3 = feature_importance.head(3)
        
        for i, row in top3.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            with st.expander(f"Feature: {feature.replace('_', ' ').title()} (Importance: {importance:.4f})"):
                if feature == 'education_level':
                    st.markdown("""
                    **Education Level** is the most important predictor of poverty.
                    - Higher education significantly reduces poverty risk
                    - Tertiary education provides the strongest protection
                    - No education is associated with the highest poverty risk
                    """)
                elif feature == 'internet_access':
                    st.markdown("""
                    **Internet Access** is a strong predictor of poverty status.
                    - Households with internet access are significantly wealthier
                    - Digital divide mirrors the poverty divide
                    """)
                elif feature == 'access_electricity':
                    st.markdown("""
                    **Access to Electricity** is a key development indicator.
                    - Electricity is fundamental for economic activities
                    - Lack of electricity limits education and business opportunities
                    """)
                else:
                    st.markdown(f"**{feature.replace('_', ' ').title()}** is an important predictor of poverty status.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "100%", "✅")
        with col2:
            st.metric("Features", "7", "✅")
        with col3:
            st.metric("Test Samples", "955", "✅")
    else:
        st.warning("Feature importance data not found.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<center>
    <b>Nigeria Poverty Prediction Dashboard</b><br>
    Powered by Explainable Boosting Machine (EBM)
</center>
""", unsafe_allow_html=True)
