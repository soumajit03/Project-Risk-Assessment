import streamlit as st
st.set_page_config(
    page_title="Project AI | Project Risk Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import os
from dotenv import load_dotenv


from agents.market_analysis import market_analysis_agent
from agents.risk_scoring import risk_scoring_agent
from agents.project_status import project_status_agent
from agents.reporting import reporting_agent

# Load environment variables
load_dotenv()

# Initialize Groq client
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Warning: '{file_name}' not found. Default styling will be used.")

local_css("styles.css")

# Constants
COLOR_PRIMARY = "#2A5C8D"
COLOR_SECONDARY = "#4B8BBE"
COLOR_ACCENT = "#FFA500"
COLOR_DANGER = "#FF4B4B"
COLOR_WARNING = "#FFA500"
COLOR_SUCCESS = "#4BB543"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#FFFFFF"  # Change text color to white

# Set theme
sns.set_theme(style="whitegrid", palette="pastel")
plt.style.use("seaborn-v0_8")

# Data loading functions with enhanced caching and error handling
@st.cache_data
def load_market_data():
    try:
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
        
        data = {
            'Date': dates,
            'S&P500': [4200 + i * 5 + np.random.normal(0, 50) for i in range(len(dates))],
            'NASDAQ': [14000 + i * 15 + np.random.normal(0, 150) for i in range(len(dates))],
            'DJIA': [33000 + i * 10 + np.random.normal(0, 100) for i in range(len(dates))],
            'Volatility': [15 + np.random.normal(0, 3) for _ in range(len(dates))],
            'Volume': [1000000 + np.random.normal(0, 200000) for _ in range(len(dates))]
        }
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_risk_data():
    try:
        data = {
            'Asset': ['US Equities', 'EU Bonds', 'Emerging Markets', 'Commodities', 'Crypto', 'Real Estate'],
            'Current_Exposure': [35, 25, 15, 10, 5, 10],
            'Risk_Score': [65, 40, 80, 70, 95, 55],
            'Return_Potential': [75, 45, 85, 60, 90, 50],
            'Liquidity': [90, 85, 60, 70, 50, 30],
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading risk data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_project_data():
    try:
        data = {
            'Project_ID': ['PRJ001', 'PRJ002', 'PRJ003', 'PRJ004', 'PRJ005'],
            'Project_Name': ['Market Expansion', 'Risk System Upgrade', 'Trading Platform', 'Compliance Update', 'Data Integration'],
            'Start_Date': ['2025-01-10', '2025-02-15', '2025-03-01', '2025-03-20', '2025-04-01'],
            'Due_Date': ['2025-06-30', '2025-05-15', '2025-07-01', '2025-05-10', '2025-08-30'],
            'Progress': [65, 80, 45, 90, 20],
            'Resource_Risk': ['Low', 'Medium', 'High', 'Low', 'Medium'],
            'Schedule_Risk': ['Medium', 'Low', 'High', 'Low', 'High'],
            'Budget_Risk': ['Low', 'Medium', 'High', 'Low', 'Medium'],
        }
        
        df = pd.DataFrame(data)
        df['Start_Date'] = pd.to_datetime(df['Start_Date'])
        df['Due_Date'] = pd.to_datetime(df['Due_Date'])
        return df
    except Exception as e:
        st.error(f"Error loading project data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_historical_risk_alerts():
    try:
        data = {
            'Date': pd.date_range(start=datetime.now() - timedelta(days=30), periods=10, freq='3D'),
            'Alert_Type': ['Market Volatility', 'Liquidity Warning', 'Exposure Limit', 'Regulatory Change', 'Concentration Risk', 
                           'Currency Risk', 'Interest Rate Risk', 'Credit Risk', 'Operational Risk', 'Compliance Issue'],
            'Severity': ['High', 'Medium', 'High', 'Medium', 'Low', 'Medium', 'High', 'Medium', 'Low', 'High'],
            'Description': [
                'Significant market volatility detected in Asian markets',
                'Reduced liquidity in corporate bond markets',
                'Technology sector exposure exceeds policy limits',
                'New SEC regulations affecting derivatives trading',
                'Portfolio concentration in energy sector above threshold',
                'Potential FX volatility due to central bank actions',
                'Interest rate movements may affect bond holdings',
                'Increased default risk in emerging market debt',
                'Settlement delays in international transactions',
                'Compliance deadline approaching for ESG reporting'
            ],
            'Status': ['Resolved', 'Resolved', 'Pending', 'Resolved', 'Pending', 'Resolved', 'Pending', 'Resolved', 'Resolved', 'Pending'],
        }
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading historical alerts: {str(e)}")
        return pd.DataFrame()




# Sidebar navigation with improved layout
def sidebar():
    with st.sidebar:
        # Logo and header
        st.markdown("""
        <div class="sidebar-header">
            <h1 class="logo">Project AI Agents</h1>
            <p class="tagline">            <p>Cliques Risk AI Platform v2.1</p>
 Risk Management Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
    "Select Agent",
    ["Dashboard", "Market Analysis", "Risk Scoring", "Project Status", "Risk Reporting", "Cliques AI Chatbot"],
    label_visibility="collapsed",
    horizontal=False,
)
        
        st.markdown("---")
        
        # User info section
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">User Information</h3>
            <div class="user-info">
                <p><strong>User:</strong> <span>Analyst</span></p>
                <p><strong>Role:</strong> <span>Risk Manager</span></p>
                <p><strong>Last Login:</strong> <span>Today</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System status
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">System Status</h3>
            <div class="system-status">
                <p><span class="status-indicator success"></span> All systems operational</p>
                <p><span class="status-indicator success"></span> API connected</p>
                <p><span class="status-indicator success"></span> Data updated</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div class="sidebar-footer">
            <p>Cliques Risk AI Platform v2.1</p>
            <p>© 2025 Cliques Project Solutions</p>
        </div>
        """, unsafe_allow_html=True)
    
    return page

# Page functions with improved UI components
def dashboard():
    st.title("📊 Project Risk Dashboard")
    st.markdown("Monitor your financial risk exposure and market trends in real-time")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Market Risk Index", value="68", delta="+3")
    with col2:
        st.metric(label="Portfolio Health", value="82%", delta="-2%")
    with col3:
        st.metric(label="Active Projects", value="5", delta="+1")
    with col4:
        st.metric(label="Open Risk Alerts", value="4", delta="-2")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Market Overview")
        market_data = load_market_data()
        fig = px.line(market_data[-30:], x=market_data[-30:].index, y=['S&P500', 'NASDAQ', 'DJIA'],
                      labels={'value': 'Index Value', 'variable': 'Index'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Recent Alerts")
        alerts = load_historical_risk_alerts()
        recent_alerts = alerts.sort_values('Date', ascending=False).head(3)
        for _, alert in recent_alerts.iterrows():
            st.write(f"**{alert['Alert_Type']}** ({alert['Severity']}) - {alert['Date'].strftime('%Y-%m-%d')}")
            st.write(alert['Description'])
            st.write("---")

    # Bottom section
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk profile
        with st.container(border=True):
            st.subheader("📊 Risk Profile")
            
            categories = ['Market Risk', 'Credit Risk', 'Liquidity Risk', 'Operational Risk', 'Regulatory Risk']
            values = [75, 45, 60, 30, 50]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Risk Profile',
                line=dict(color=COLOR_PRIMARY)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Asset exposure
        with st.container(border=True):
            st.subheader("💼 Asset Exposure")
            risk_data = load_risk_data()
            
            fig = px.bar(risk_data, x='Asset', y='Current_Exposure',
                         color='Risk_Score', color_continuous_scale='Bluered',
                         labels={'Current_Exposure': 'Exposure (%)', 'Risk_Score': 'Risk Score'},
                         text='Current_Exposure')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

def market_analysis_page():
    st.title("🔍 Market Analysis Agent")
    st.markdown("Analyze financial trends and news with AI-powered insights")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("📝 Analysis Request")
            
            analysis_result = None  # Initialize result variable
            
            with st.form("market_analysis_form"):
                analysis_query = st.text_area(
                    "Enter your market analysis query",
                    height=150,
                    placeholder="Example: Analyze recent trends in tech stocks and how they might be affected by current interest rate policies",
                    help="Be specific about the assets, timeframes, and factors you want analyzed"
                )
                
                submitted = st.form_submit_button("Generate Analysis", type="primary", use_container_width=True)
                
                if submitted and analysis_query:
                    if len(analysis_query.strip()) < 10:
                        st.warning("Please enter a more detailed query (at least 10 characters)")
                    else:
                        with st.spinner("Generating market analysis..."):
                            analysis_result = market_analysis_agent(analysis_query)
            
            # Display the result outside the form
            if analysis_result:
                with st.container(border=True):
                    st.subheader("📋 Analysis Results")
                    st.markdown(analysis_result)
    
    with col2:
        with st.container(border=True):
            st.subheader("ℹ️ Market Data")
            
            market_data = load_market_data()
            latest_data = market_data.iloc[-1]
            
            st.metric("S&P 500", f"{latest_data['S&P500']:,.2f}")
            st.metric("NASDAQ", f"{latest_data['NASDAQ']:,.2f}")
            st.metric("DJIA", f"{latest_data['DJIA']:,.2f}")
            st.metric("Volatility", f"{latest_data['Volatility']:.2f}")
            
            st.markdown("---")
            
            st.write("**Recent Trends**")
            st.write(f"📈 5-day change: {((market_data['S&P500'].iloc[-1] - market_data['S&P500'].iloc[-5]) / market_data['S&P500'].iloc[-5] * 100):.2f}%")
            st.write(f"📉 30-day change: {((market_data['S&P500'].iloc[-1] - market_data['S&P500'].iloc[-30]) / market_data['S&P500'].iloc[-30] * 100):.2f}%")

    # Market visualizations
    with st.container(border=True):
        st.subheader("📊 Market Visualizations")
        
        market_data = load_market_data()
        tab1, tab2, tab3 = st.tabs(["Index Performance", "Volatility", "Correlation"])
        
        with tab1:
            fig = px.line(market_data[-90:], x=market_data[-90:].index, y=['S&P500', 'NASDAQ', 'DJIA'],
                          labels={'value': 'Index Value', 'variable': 'Index'},
                          color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig = px.line(market_data[-90:], x=market_data[-90:].index, y='Volatility',
                          labels={'value': 'Volatility Index', 'variable': 'Volatility'},
                          color_discrete_sequence=[COLOR_PRIMARY])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            corr = market_data[['S&P500', 'NASDAQ', 'DJIA']].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

def risk_scoring_page():
    st.title("📉 Risk Scoring Agent")
    st.markdown("Assess transaction and investment risks with AI-powered analysis")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("📝 Risk Assessment")
            
            risk_result = None  # Initialize result variable
            
            with st.form("risk_scoring_form"):
                asset_options = ['Equities', 'Bonds', 'Real Estate', 'Commodities', 'Cryptocurrency', 
                                 'Derivatives', 'Foreign Exchange', 'Private Equity']
                asset_type = st.selectbox("Select Asset Type", asset_options)
                
                risk_query = st.text_area(
                    "Enter your risk assessment query",
                    height=150,
                    placeholder="Example: Evaluate the risk profile of investing in tech sector equities given current market conditions",
                    help="Describe the specific investment or transaction you want assessed"
                )
                
                submitted = st.form_submit_button("Assess Risk", type="primary", use_container_width=True)
                
                if submitted and risk_query:
                    if len(risk_query.strip()) < 10:
                        st.warning("Please enter a more detailed query (at least 10 characters)")
                    else:
                        with st.spinner("Generating risk assessment..."):
                            risk_result = risk_scoring_agent(asset_type, risk_query)
            
            # Display the result outside the form
            if risk_result:
                with st.container(border=True):
                    st.subheader("📋 Risk Assessment Results")
                    st.markdown(risk_result)
    
    with col2:
        with st.container(border=True):
            st.subheader("📊 Risk Profile")
            
            risk_data = load_risk_data()
            selected_asset_idx = min(asset_options.index(asset_type) if asset_type in asset_options else 0, len(risk_data) - 1)
            
            if selected_asset_idx < len(risk_data):
                risk_score = risk_data.iloc[selected_asset_idx]['Risk_Score']
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'steps': [
                            {'range': [0, 40], 'color': COLOR_SUCCESS},
                            {'range': [40, 70], 'color': COLOR_WARNING},
                            {'range': [70, 100], 'color': COLOR_DANGER}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score}
                    }
                ))
                
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Risk Level:** {'Low' if risk_score < 40 else 'Medium' if risk_score < 70 else 'High'}")
                st.write(f"**Recommended Action:** {'Monitor' if risk_score < 40 else 'Review' if risk_score < 70 else 'Mitigate'}")

    # Risk data visualization
    with st.container(border=True):
        st.subheader("📈 Asset Risk Overview")
        
        risk_data = load_risk_data()
        fig = px.scatter(risk_data, x='Risk_Score', y='Return_Potential', 
                         size='Current_Exposure', color='Asset',
                         hover_name='Asset', size_max=30,
                         labels={'Risk_Score': 'Risk Score', 'Return_Potential': 'Return Potential'},
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(risk_data, use_container_width=True, hide_index=True)

def project_status_page():
    st.title("📅 Project Status Agent")
    st.markdown("Track project progress and internal risks with AI-powered insights")
    
    # Project data
    project_data = load_project_data()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("📋 Project Overview")
            
            selected_project = st.selectbox(
                "Select Project",
                options=project_data['Project_Name'].tolist(),
                key="project_select"
            )
            
            project_info = project_data[project_data['Project_Name'] == selected_project].iloc[0]
            
            # Project details
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Progress", f"{project_info['Progress']}%")
                st.progress(project_info['Progress']/100)
                
            with col_b:
                days_remaining = (project_info['Due_Date'] - datetime.now()).days
                st.metric("Days Remaining", days_remaining)
            
            # Timeline visualization
            start = project_info['Start_Date']
            end = project_info['Due_Date']
            today = pd.Timestamp(datetime.now().date())
            
            total_days = (end - start).days
            elapsed_days = (today - start).days
            
            timeline_progress = min(max(elapsed_days / total_days, 0), 1)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = timeline_progress * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Timeline Progress"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 33], 'color': COLOR_DANGER},
                        {'range': [33, 66], 'color': COLOR_WARNING},
                        {'range': [66, 100], 'color': COLOR_SUCCESS}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': timeline_progress * 100}
                }
            ))
            
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            
            risk_cols = st.columns(3)
            with risk_cols[0]:
                resource_risk = project_info['Resource_Risk']
                color = COLOR_SUCCESS if resource_risk == "Low" else COLOR_WARNING if resource_risk == "Medium" else COLOR_DANGER
                st.markdown(f"**Resource Risk:** <span style='color:{color}'>{resource_risk}</span>", unsafe_allow_html=True)
                
            with risk_cols[1]:
                schedule_risk = project_info['Schedule_Risk']
                color = COLOR_SUCCESS if schedule_risk == "Low" else COLOR_WARNING if schedule_risk == "Medium" else COLOR_DANGER
                st.markdown(f"**Schedule Risk:** <span style='color:{color}'>{schedule_risk}</span>", unsafe_allow_html=True)
                
            with risk_cols[2]:
                budget_risk = project_info['Budget_Risk']
                color = COLOR_SUCCESS if budget_risk == "Low" else COLOR_WARNING if budget_risk == "Medium" else COLOR_DANGER
                st.markdown(f"**Budget Risk:** <span style='color:{color}'>{budget_risk}</span>", unsafe_allow_html=True)
            
            # Status analysis
            with st.form("project_status_form"):
                context = st.text_area(
                    "Enter additional context for project analysis",
                    height=100,
                    placeholder="Example: The project team has reported potential delays in the integration phase due to vendor API changes"
                )
                
                submitted = st.form_submit_button("Analyze Project Status", type="primary", use_container_width=True)
                
                if submitted:
                    with st.spinner("Analyzing project status..."):
                        status_result = project_status_agent(selected_project, context)
                        if status_result:
                            with st.container(border=True):
                                st.subheader("📋 Status Analysis")
                                st.markdown(status_result)
    
    with col2:
        with st.container(border=True):
            st.subheader("📌 All Projects")
            
            for idx, row in project_data.iterrows():
                with st.container(border=True):
                    st.write(f"**{row['Project_Name']}**")
                    
                    if row['Progress'] < 25:
                        status = "🔴 At Risk" if row['Schedule_Risk'] == "High" else "🟡 Early Stage" 
                    elif row['Progress'] < 75:
                        status = "🔴 At Risk" if row['Schedule_Risk'] == "High" else "🟢 On Track"
                    else:
                        status = "🟡 Final Stage" if row['Schedule_Risk'] == "High" else "🟢 Near Completion"
                    
                    st.progress(row['Progress']/100)
                    st.write(f"**Status:** {status}")
                    st.write(f"**Due:** {row['Due_Date'].strftime('%Y-%m-%d')}")

    # Project timeline visualization
    with st.container(border=True):
        st.subheader("📅 Project Timeline")
        
        fig = px.timeline(
            project_data, 
            x_start="Start_Date", 
            x_end="Due_Date", 
            y="Project_Name",
            color="Progress",
            color_continuous_scale='Bluered',
            labels={'Project_Name': 'Project', 'Start_Date': 'Start Date', 'Due_Date': 'Due Date'}
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def reporting_page():


    st.title("📑 Risk Reporting Agent")
    st.markdown("Generate detailed risk analytics and alerts with AI-powered reporting")
    
    # Report generation
    with st.container(border=True):
        st.subheader("📝 Report Generator")
        
        with st.form("reporting_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Report Type",
                    options=["Comprehensive Risk Report", "Market Risk Analysis", "Portfolio Risk Assessment", 
                             "Project Risk Report", "Regulatory Compliance Report", "Custom Report"],
                    help="Select the type of report you need"
                )
                
                timeframe = st.selectbox(
                    "Timeframe",
                    options=["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Custom"],
                    help="Select the reporting timeframe"
                )
                
                if timeframe == "Custom":
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
                    end_date = st.date_input("End Date", value=datetime.now())
                    timeframe = f"Custom ({start_date} to {end_date})"
            
            with col2:
                details = st.text_area(
                    "Additional Details",
                    height=150,
                    placeholder="Specify any particular focus areas, risk thresholds, or specific assets to include in the report",
                    help="Add any specific requirements for the report"
                )
            
            submitted = st.form_submit_button("Generate Report", type="primary", use_container_width=True)
            
            if submitted:
                with st.spinner("Generating risk report..."):
                    report_result = reporting_agent(report_type, timeframe, details)
                    if report_result:
                        with st.container(border=True):
                            st.subheader("📋 Generated Report")
                            st.markdown(report_result)
    
    # Historical alerts
    with st.container(border=True):
        st.subheader("⚠️ Historical Risk Alerts")
        
        alerts = load_historical_risk_alerts()
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
                key="severity_filter"
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=["Pending", "Resolved"],
                default=["Pending", "Resolved"],
                key="status_filter"
            )
        
        # Apply filters
        filtered_alerts = alerts[
            alerts['Severity'].isin(severity_filter) &
            alerts['Status'].isin(status_filter)
        ]
        
        # Display alerts
        if not filtered_alerts.empty:
            for _, alert in filtered_alerts.iterrows():
                severity_color = COLOR_DANGER if alert['Severity'] == "High" else COLOR_WARNING if alert['Severity'] == "Medium" else COLOR_ACCENT
                
                with st.container(border=True):
                    st.markdown(f"""
                    <div class="alert-card" style="border-left-color: {severity_color}">
                        <p class="alert-title">{alert['Alert_Type']}</p>
                        <p class="alert-date">{alert['Date'].strftime('%Y-%m-%d')}</p>
                        <p class="alert-description">{alert['Description']}</p>
                        <p class="alert-status">Status: <span style="color: {'green' if alert['Status'] == 'Resolved' else 'orange'};">{alert['Status']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No alerts match the selected filters.")

def crew_ai_page():
    st.title("🤝 Cliques AI Chatbot")
    st.markdown("Collaborate with all agents through a unified chatbot interface.")

    # Input fields for the chatbot
    with st.form("crew_ai_form"):
        query = st.text_area(
            "Enter your query",
            height=150,
            placeholder="Example: Analyze the risk of investing in tech stocks and provide a project status update.",
        )
        asset_type = st.selectbox(
            "Select Asset Type (optional)",
            options=["Equities", "Bonds", "Real Estate", "Commodities", "Cryptocurrency", "Derivatives", "Foreign Exchange", "Private Equity"],
            index=0,
        )
        project_name = st.text_input("Project Name (optional)", placeholder="Example: Market Expansion")
        report_type = st.selectbox(
            "Report Type (optional)",
            options=["Comprehensive Risk Report", "Market Risk Analysis", "Portfolio Risk Assessment", "Project Risk Report", "Regulatory Compliance Report", "Custom Report"],
            index=0,
        )
        timeframe = st.selectbox(
            "Timeframe (optional)",
            options=["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Custom"],
            index=0,
        )
        details = st.text_area(
            "Additional Details (optional)",
            height=100,
            placeholder="Specify any particular focus areas, risk thresholds, or specific assets to include in the report.",
        )

        submitted = st.form_submit_button("Submit Query")

        if submitted:
            with st.spinner("Cliques AI is processing your query..."):
                from agents.crew_ai import crew_ai_agent
                response = crew_ai_agent(query, asset_type, project_name, report_type, timeframe, details)
                if response:
                    st.subheader("📋 Cliques AI Response")
                    st.markdown(response)
                else:
                    st.error("No response received. Please check your query or try again.")


# Main application logic
def main():
    page = sidebar()
    
    if page == "Dashboard":
        dashboard()
    elif page == "Market Analysis":
        market_analysis_page()
    elif page == "Risk Scoring":
        risk_scoring_page()
    elif page == "Project Status":
        project_status_page()
    elif page == "Risk Reporting":
        reporting_page()
    elif page == "Cliques AI Chatbot":
        crew_ai_page()

if __name__ == "__main__":
    main()