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

def project_status_agent(project_name, context):
    try:
        prompt = f"""
        You are a Project Status Tracking Agent specialized in monitoring project progress and internal risks.
        Analyze the following project and context to provide a status assessment:
        
        PROJECT NAME: {project_name}
        CONTEXT: {context}
        
        Please include:
        1. Current project health assessment
        2. Identified internal risks (resource constraints, schedule delays, etc.)
        3. Risk mitigation strategies
        4. Progress evaluation
        5. Recommendations for keeping the project on track
        """
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a project management expert providing detailed project status assessment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error in project status assessment: {str(e)}")
        return None

