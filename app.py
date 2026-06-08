import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Web Search ---
def search_web(query, max_results=6):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results

# --- Format search results ---
def format_results(results):
    formatted = ""
    for r in results:
        formatted += f"Title: {r.get('title', '')}\n"
        formatted += f"Summary: {r.get('body', '')}\n\n"
    return formatted

# --- Generate report with Groq ---
def generate_report(topic, search_data):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert market research analyst.
Using the search data provided, generate a comprehensive market research report.
Structure it with these sections:

## 📊 Market Overview
## 🏢 Key Players & Competitors
## 📈 Current Trends
## ⚠️ Challenges & Risks
## 💡 Opportunities
## 🎯 Strategic Recommendations

Be specific, data-driven, and concise. Use bullet points within sections.

Search Data:
{search_data}"""),
        ("human", "Generate a market research report for: {topic}")
    ])
    chain = prompt | llm
    response = chain.invoke({"search_data": search_data, "topic": topic})
    return response.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Market Research", page_icon="📊")
st.title("📊 AI Market Research Platform")
st.caption("Enter any company, industry or topic to get an instant research report")

topic = st.text_input(
    "Research Topic:",
    placeholder="e.g. Generative AI market in India, Tesla competitors, EdTech trends 2025"
)

col1, col2 = st.columns(2)
with col1:
    search_queries = st.number_input("Search queries to run", min_value=1, max_value=5, value=3)
with col2:
    results_per_query = st.number_input("Results per query", min_value=3, max_value=8, value=5)

if st.button("🔍 Generate Report") and topic:
    # Build search queries
    queries = [
        f"{topic} market overview 2025",
        f"{topic} competitors and key players",
        f"{topic} trends and opportunities",
        f"{topic} challenges and risks",
        f"{topic} market size revenue"
    ][:search_queries]

    all_results = ""
    with st.spinner("🔍 Searching the web..."):
        progress = st.progress(0)
        for i, query in enumerate(queries):
            results = search_web(query, max_results=results_per_query)
            all_results += f"\n--- Query: {query} ---\n"
            all_results += format_results(results)
            progress.progress((i + 1) / len(queries))

    st.success(f"✅ Gathered {len(queries) * results_per_query} data points")

    with st.spinner("🧠 Generating report with AI..."):
        report = generate_report(topic, all_results)

    st.markdown("---")
    st.markdown(report)

    with st.expander("🔍 Raw Search Data"):
        st.text(all_results[:3000] + "..." if len(all_results) > 3000 else all_results)

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"market_research_{topic[:30].replace(' ', '_')}.txt",
        mime="text/plain"
    )