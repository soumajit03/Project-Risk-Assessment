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

def market_analysis_agent(query):
    try:
        if not query or len(query.strip()) < 10:
            raise ValueError("Query must be at least 10 characters long")
            
        prompt = f"""
        You are a Market Analysis Agent specializing in financial trends and news analysis.
        Analyze the following query and provide expert insights, relevant trends, and financial implications:
        
        QUERY: {query}
        
        Please include:
        1. Key market insights related to the query
        2. Potential impacts on investments
        3. Related news/trends that might influence decisions
        4. A balanced risk assessment
        """
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a financial market analysis expert providing insights on market trends and news."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error in market analysis: {str(e)}")
        return None
