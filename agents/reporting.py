import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    groq_client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    groq_client = None  # Prevent further errors if initialization fails


def reporting_agent(report_type, timeframe, details):
    try:
        prompt = f"""
        You are a Reporting Agent specialized in providing detailed risk analytics and alerts.
        Generate a {report_type} report based on the following parameters:
        
        TIMEFRAME: {timeframe}
        DETAILS: {details}
        
        Please include:
        1. Executive summary
        2. Key risk metrics
        3. Notable trends or patterns
        4. Alert thresholds and triggers
        5. Recommended actions
        """
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a financial reporting expert providing detailed risk analytics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error in report generation: {str(e)}")
        return None