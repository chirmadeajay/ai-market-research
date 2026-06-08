import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Generate report with Groq ---
def generate_report(topic, report_type):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert market research analyst with deep knowledge across industries.
Generate a comprehensive, data-rich market research report.
Structure it with these sections:

## 📊 Market Overview
## 🏢 Key Players & Competitors
## 📈 Current Trends (2024-2025)
## ⚠️ Challenges & Risks
## 💡 Opportunities
## 🎯 Strategic Recommendations

Be specific with company names, market sizes, percentages, and real examples.
Use bullet points within sections. Be professional and concise."""),
        ("human", "Generate a {report_type} market research report for: {topic}")
    ])
    chain = prompt | llm
    response = chain.invoke({"topic": topic, "report_type": report_type})
    return response.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Market Research", page_icon="📊")
st.title("📊 AI Market Research Platform")
st.caption("Enter any company, industry or topic to get an instant AI-powered research report")

topic = st.text_input(
    "Research Topic:",
    placeholder="e.g. Generative AI market in India, Tesla, EdTech trends 2025"
)

report_type = st.selectbox(
    "Report Type:",
    ["comprehensive", "competitive analysis", "trend analysis", "investment", "startup landscape"]
)

if st.button("🔍 Generate Report") and topic:
    with st.spinner("🧠 Analysing and generating report..."):
        report = generate_report(topic, report_type)

    st.markdown("---")
    st.markdown(report)

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"market_research_{topic[:30].replace(' ', '_')}.txt",
        mime="text/plain"
    )