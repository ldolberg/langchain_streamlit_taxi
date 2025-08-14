import os
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI


# App config
st.set_page_config(page_title="Taxi SQL Chat (Gemini + LangChain)", page_icon="🚕", layout="wide")

st.title("🚕 Taxi SQL Chat — LangChain + Gemini + Postgres")
st.caption("Ask natural language questions; the app generates SQL for Postgres using Google Gemini and runs it against a small Taxi dataset.")

# Environment & connections
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/taxi")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Google API Key (Gemini)", value=GOOGLE_API_KEY, type="password", help="If not set as env var, paste it here."
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        GOOGLE_API_KEY = api_key_input

    st.write("Database URL:")
    st.code(DATABASE_URL)

    if st.button("Clear history"):
        st.session_state.pop("history", None)

    hint = st.text_area(
        "Hint for the model (optional)",
        placeholder="e.g., Use pickup_datetime for date filters; always LIMIT 50; group by day",
        help="Your guidance will be added to the prompts used for generating or fixing SQL."
    )

if not GOOGLE_API_KEY:
    st.info("Provide your Google API key in the sidebar to start.")

# Initialize connections lazily
@st.cache_resource(show_spinner=False)
def get_db_objects():
    engine = create_engine(DATABASE_URL)
    db = SQLDatabase.from_uri(DATABASE_URL)
    return engine, db

@st.cache_resource(show_spinner=False)
def get_llm():
    # Use the lightweight, fast model for SQL generation
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)


def build_sql_prompt(question: str, schema: str, hint: str | None = None) -> str:
    return (
        "You are an expert data analyst and PostgreSQL SQL generator. "
        "Given a question and a database schema, write a syntactically correct and executable SQL query for PostgreSQL that answers the question.\n\n"
        f"Schema:\n{schema}\n\n"
        + (f"User hint:\n{hint}\n\n" if hint else "")
        "Guidelines:\n"
        "- Only use the tables and columns shown in the schema.\n"
        "- Prefer aggregate queries for counting/summing/averages.\n"
        "- If many rows would be returned, include 'ORDER BY' where sensible and add 'LIMIT 50'.\n"
        "- Use single quotes for string literals.\n"
        "- Timestamps are in UTC; filter by timestamp columns if needed.\n"
        "- Do NOT add explanations. Output ONLY the SQL, with no backticks.\n\n"
        f"Question: {question}\nSQL:"
    )


def sanitize_sql_text(text_value: str) -> str:
    # Remove markdown fences if model adds them
    cleaned = re.sub(r"```(sql)?", "", text_value, flags=re.IGNORECASE).strip()
    # Ensure ends with semicolon
    if not cleaned.rstrip().endswith(";"):
        cleaned = cleaned.rstrip() + ";"
    return cleaned


def build_fix_sql_prompt(question: str, schema: str, bad_sql: str, error_message: str, hint: str | None = None) -> str:
    return (
        "You wrote a PostgreSQL SQL query that failed to execute. Fix it.\n\n"
        f"Schema:\n{schema}\n\n"
        + (f"User hint:\n{hint}\n\n" if hint else "")
        "Original question:\n"
        f"{question}\n\n"
        "Previous SQL (may be wrong):\n"
        f"{bad_sql}\n\n"
        "Database error message:\n"
        f"{error_message}\n\n"
        "Return ONLY a corrected SQL statement that will run on PostgreSQL. No explanations, no backticks."
    )


# UI input
question = st.text_input("Ask a question about the yellow taxi trips dataset:", placeholder="e.g., What was the average trip distance by day in January 2019?")
submit = st.button("Ask")

if submit and question:
    if not GOOGLE_API_KEY:
        st.error("Please provide GOOGLE_API_KEY in the sidebar.")
        st.stop()

    engine, db = get_db_objects()
    llm = get_llm()

    with st.spinner("Reading schema and generating SQL with Gemini..."):
        schema_info = db.get_table_info()
        prompt = build_sql_prompt(question, schema_info, hint)
        sql_text = llm.invoke(prompt).content
        sql_query = sanitize_sql_text(sql_text)

    st.subheader("Generated SQL")
    st.code(sql_query, language="sql")

    # Try running the SQL; if it fails, ask Gemini to fix it up to 2 times
    df = None
    sql_to_run = sql_query
    last_error: str | None = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(text(sql_to_run), conn)
            break
        except Exception as e:
            last_error = str(e)
            if attempt >= max_attempts:
                st.error(f"Query failed after {max_attempts} attempts: {last_error}")
                break
            with st.spinner(f"SQL failed (attempt {attempt}). Asking Gemini to fix and retry..."):
                fix_prompt = build_fix_sql_prompt(question, schema_info, sql_to_run, last_error, hint)
                fixed_sql_text = llm.invoke(fix_prompt).content
                sql_to_run = sanitize_sql_text(fixed_sql_text)
                st.subheader(f"Corrected SQL (attempt {attempt + 1})")
                st.code(sql_to_run, language="sql")

    if df is not None:
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Simple natural language summary (optional)
        try:
            summary_prompt = (
                "You are a helpful data analyst. Summarize the result for a non-technical user in 1-3 sentences.\n\n"
                f"Question: {question}\n"
                f"SQL: {sql_query}\n"
                f"First 20 rows of results as CSV:\n{df.head(20).to_csv(index=False)}\n"
            )
            nl_answer = llm.invoke(summary_prompt).content
            st.subheader("Answer")
            st.write(nl_answer)
        except Exception:
            pass

        # Save to history
        history: List[Dict] = st.session_state.get("history", [])
        history.append({"question": question, "sql": sql_query, "rows": len(df)})
        st.session_state["history"] = history

if "history" in st.session_state and st.session_state["history"]:
    st.divider()
    st.subheader("History (this session)")
    for i, item in enumerate(reversed(st.session_state["history"])):
        st.markdown(f"**Q:** {item['question']}")
        with st.expander("See generated SQL"):
            st.code(item["sql"], language="sql")
        st.caption(f"Rows returned: {item['rows']}")
