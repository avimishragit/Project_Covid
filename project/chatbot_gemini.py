"""
chatbot_gemini.py
A Streamlit-compatible chatbot script using Google Gemini LLM via LangChain.
This script provides a function to get answers from Gemini and can be integrated into your Streamlit app.
"""
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Import the DuckDuckGo web search agent
from project.web_search_agent import web_search

# Set your Gemini API key (ensure this is set in your environment for security)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini LLM via LangChain
@st.cache_resource
def get_gemini_llm():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
        max_output_tokens=2048,
    )

def ask_gemini(question, context=None):
    """
    Query Gemini LLM with a user question and optional context. If Gemini cannot answer, fallback to DuckDuckGo web search.
    Args:
        question (str): The user's question.
        context (str, optional): Additional context or data to provide to the LLM.
    Returns:
        str: The LLM's answer or web search result.
    """
    # System prompt with project and data context
    system_prompt = (
        "You are a helpful COVID-19 data assistant for a time series forecasting and visualization dashboard. "
        "You have access to global COVID-19 data with the following columns: Province/State, Country/Region, Lat, Long, Date, Confirmed, Deaths, Recovered, Active, WHO Region. "
        "The data covers all countries and regions from 2020-01-22 onward. "
        "You can answer questions about COVID-19 trends, forecasts, data columns, project features, and modeling approaches. "
        "If you do not know the answer, say so, and a web search will be performed."
    )
    # Optionally add a sample of the data for context
    data_sample = ""
    try:
        import pandas as pd
        df = pd.read_csv("../Data_original/covid.csv", nrows=3)
        data_sample = f"\nHere is a sample of the data (first 3 rows):\n{df.to_markdown(index=False)}\n"
    except Exception:
        pass
    full_context = (context or "") + system_prompt + data_sample
    llm = get_gemini_llm()
    messages = [
        SystemMessage(content=full_context),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, 'content') else str(response)
    fallback_phrases = [
        "I don't know", "I'm not sure", "cannot answer", "don't have information", "no information", "Sorry", "as an AI language model"
    ]
    if any(phrase.lower() in answer.lower() for phrase in fallback_phrases) or len(answer.strip()) < 10:
        web_result = web_search(question)
        return f"[Gemini]: {answer}\n\n[Web Search]: {web_result}"
    return answer

# Example Streamlit UI usage:
if __name__ == "__main__":
    st.title("Gemini Chatbot Demo")
    user_q = st.text_input("Ask a question:")
    if user_q:
        with st.spinner("Gemini is thinking..."):
            try:
                answer = ask_gemini(user_q)
                st.success(answer)
            except Exception as e:
                st.error(f"Error: {e}")
