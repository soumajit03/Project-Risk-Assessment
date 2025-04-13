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

def risk_scoring_agent(asset_type, query):
    try:
        if not query or len(query.strip()) < 10:
            raise ValueError("Query must be at least 10 characters long")
            
        prompt = f"""
        You are a Risk Scoring Agent specializing in transaction and investment risk assessment.
        Analyze the following asset type and query to provide a detailed risk assessment:
        
        ASSET TYPE: {asset_type}
        QUERY: {query}
        
        Please include:
        1. Overall risk score (1-100)
        2. Primary risk factors
        3. Risk mitigation recommendations
        4. Market conditions affecting risk
        5. Short-term and long-term risk outlook
        """
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a financial risk assessment expert providing detailed risk analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error in risk scoring: {str(e)}")
        return None