import os
import re
import time
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
import altair as alt

try:
    # Optional import for in-app ETL
    from etl.load_tlc import load_month as etl_load_month, ensure_schema as etl_ensure_schema
except Exception:
    etl_load_month = None
    etl_ensure_schema = None

try:
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:
    google_exceptions = None  # type: ignore


# App config
st.set_page_config(
    page_title="Taxi SQL Chat (Gemini + LangChain)",
    page_icon="🚕",
    layout="wide",
)

st.title("🚕 Taxi SQL Chat — LangChain + Gemini + Postgres")
st.caption(
    "Ask natural language questions; the app generates SQL for Postgres using "
    "Google Gemini and runs it against a small Taxi dataset."
)

# Environment & connections
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/taxi")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Google API Key (Gemini)",
        value=GOOGLE_API_KEY,
        type="password",
        help="If not set as env var, paste it here.",
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

    st.subheader("LLM")
    model = st.selectbox("Model", options=["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    st.subheader("Query defaults")
    default_limit = st.number_input(
        "Default LIMIT", min_value=10, max_value=1000, value=50, step=10
    )

    st.subheader("Data loader (NYC TLC)")
    month = st.text_input("Month (YYYY-MM)", value="2019-01")
    etl_limit = st.number_input(
        "Rows to load", min_value=1000, max_value=200000, value=20000, step=1000
    )
    if st.button("Load month into Postgres"):
        if etl_load_month is None:
            st.error("ETL module not available in this build.")
        else:
            try:
                engine = create_engine(DATABASE_URL)
                with st.spinner(
                    f"Loading NYC TLC {month} ({etl_limit} rows)..."
                ):
                    etl_ensure_schema(engine)  # type: ignore
                    inserted = etl_load_month(engine, month, int(etl_limit))  # type: ignore
                st.success(f"Inserted {inserted} rows from {month}.")
            except Exception as etl_err:
                st.error(f"ETL failed: {etl_err}")

if not GOOGLE_API_KEY:
    st.info("Provide your Google API key in the sidebar to start.")

# Initialize connections lazily
@st.cache_resource(show_spinner=False)
def get_db_objects():
    engine = create_engine(DATABASE_URL)
    db = SQLDatabase.from_uri(DATABASE_URL)
    return engine, db

@st.cache_resource(show_spinner=False)
def get_llm(selected_model: str, temp: float):
    return ChatGoogleGenerativeAI(model=selected_model, temperature=float(temp))


def build_sql_prompt(question: str, schema: str, hint: Optional[str] = None, default_limit: int = 50) -> str:
    return (
        "You are an expert data analyst and PostgreSQL SQL generator. "
        "Given a question and a database schema, write a syntactically correct and executable SQL query for PostgreSQL that answers the question.\n\n"
        f"Schema:\n{schema}\n\n"
        + (f"User hint:\n{hint}\n\n" if hint else "")
        + "Guidelines:\n"
        "- Only use the tables and columns shown in the schema.\n"
        "- Prefer aggregate queries for counting/summing/averages.\n"
        f"- If many rows would be returned, include 'ORDER BY' where sensible and add 'LIMIT {default_limit}'.\n"
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


def is_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    keywords = [
        "quota",
        "exceeded",
        "rate limit",
        "too many requests",
        "resource exhausted",
        "429",
    ]
    if any(k in msg for k in keywords):
        return True
    if google_exceptions is not None:
        quota_types = tuple(
            t
            for t in [
                getattr(google_exceptions, "ResourceExhausted", None),
                getattr(google_exceptions, "TooManyRequests", None),
            ]
            if t is not None
        )
        if quota_types and isinstance(err, quota_types):
            return True
    return False


def build_fix_sql_prompt(question: str, schema: str, bad_sql: str, error_message: str, hint: Optional[str] = None, default_limit: int = 50) -> str:
    return (
        "You wrote a PostgreSQL SQL query that failed to execute. Fix it.\n\n"
        f"Schema:\n{schema}\n\n"
        + (f"User hint:\n{hint}\n\n" if hint else "")
        + (f"Default LIMIT to apply when appropriate: {default_limit}.\n\n" if default_limit else "")
        + "Original question:\n"
        f"{question}\n\n"
        "Previous SQL (may be wrong):\n"
        f"{bad_sql}\n\n"
        "Database error message:\n"
        f"{error_message}\n\n"
        "Return ONLY a corrected SQL statement that will run on PostgreSQL. No explanations, no backticks."
    )


@st.cache_data(show_spinner=False)
def get_schema_df(db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    query = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name='yellow_trips'
    ORDER BY ordinal_position;
    """
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn)


@st.cache_data(show_spinner=False)
def preview_table(db_url: str, limit: int = 50) -> pd.DataFrame:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return pd.read_sql_query(
            text(
                f"SELECT * FROM yellow_trips ORDER BY pickup_datetime DESC LIMIT {int(limit)}"
            ),
            conn,
        )


def render_charts(df: pd.DataFrame):
    try:
        time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])]

        if time_cols and num_cols:
            x_col = time_cols[0]
            y_col = num_cols[0]
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(x=alt.X(x_col, title=x_col), y=alt.Y(y_col, title=y_col), tooltip=list(df.columns))
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
            return
        if cat_cols and num_cols:
            x_col = cat_cols[0]
            y_col = num_cols[0]
            # For large cardinality, show top 30
            top = (
                df.groupby(x_col)[y_col]
                .sum()
                .reset_index()
                .sort_values(y_col, ascending=False)
                .head(30)
            )
            chart = (
                alt.Chart(top)
                .mark_bar()
                .encode(x=alt.X(x_col, sort='-y'), y=alt.Y(y_col), tooltip=list(top.columns))
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass


# UI input
st.write("")
example_cols = st.columns(3)
examples = [
    "Average trip distance by day in January 2019",
    "Top 10 pickup locations by total tips",
    "Hourly trip counts for weekdays vs weekends",
]
for i, ex in enumerate(examples):
    if example_cols[i].button(ex):
        st.session_state["prefill_q"] = ex

question = st.text_input(
    "Ask a question about the yellow taxi trips dataset:",
    value=st.session_state.get("prefill_q", ""),
    placeholder="e.g., What was the average trip distance by day in January 2019?",
)
submit = st.button("Ask")

if submit and question:
    if not GOOGLE_API_KEY:
        st.error("Please provide GOOGLE_API_KEY in the sidebar.")
        st.stop()

    engine, db = get_db_objects()
    llm = get_llm(model, temperature)

    with st.spinner("Reading schema and generating SQL with Gemini..."):
        schema_info = db.get_table_info()
        prompt = build_sql_prompt(question, schema_info, hint, default_limit)
        try:
            sql_text = llm.invoke(prompt).content
        except Exception as e:
            if is_quota_error(e):
                st.error(f"Gemini quota exceeded or rate-limited. Please wait and try again. Details: {e}")
                st.stop()
            raise
        sql_query = sanitize_sql_text(sql_text)

    st.subheader("Generated SQL")
    st.code(sql_query, language="sql")

    # Try running the SQL; if it fails, ask Gemini to fix it up to 2 times
    df = None
    sql_to_run = sql_query
    last_error: str | None = None
    max_attempts = 3
    start = time.time()
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
                fix_prompt = build_fix_sql_prompt(question, schema_info, sql_to_run, last_error, hint, default_limit)
                try:
                    fixed_sql_text = llm.invoke(fix_prompt).content
                except Exception as e2:
                    if is_quota_error(e2):
                        st.error(f"Gemini quota exceeded or rate-limited during fix. Please wait and try again. Details: {e2}")
                        st.stop()
                    raise
                sql_to_run = sanitize_sql_text(fixed_sql_text)
                st.subheader(f"Corrected SQL (attempt {attempt + 1})")
                st.code(sql_to_run, language="sql")
    elapsed_ms = int((time.time() - start) * 1000)

    if df is not None:
        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", len(df))
        m2.metric("Elapsed", f"{elapsed_ms} ms")
        m3.metric("Columns", df.shape[1])

        tabs = st.tabs(["Data", "Chart", "SQL", "Summary", "Download"]) 
        with tabs[0]:
            st.dataframe(df, use_container_width=True)
        with tabs[1]:
            render_charts(df)
        with tabs[2]:
            st.code(sql_to_run, language="sql")
            explain = st.checkbox(
                "Show EXPLAIN plan", value=False, key="explain"
            )
            if explain:
                try:
                    with engine.connect() as conn:
                        plan = pd.read_sql_query(text("EXPLAIN " + sql_to_run), conn)
                    st.dataframe(plan, use_container_width=True)
                except Exception as _:
                    st.info("EXPLAIN not available for this statement.")
        with tabs[3]:
            # Simple natural language summary (optional)
            try:
                summary_prompt = (
                    "You are a helpful data analyst. Summarize the result for a non-technical user in 1-3 sentences.\n\n"
                    f"Question: {question}\n"
                    f"SQL: {sql_to_run}\n"
                    f"First 20 rows of results as CSV:\n{df.head(20).to_csv(index=False)}\n"
                )
                nl_answer = llm.invoke(summary_prompt).content
                st.write(nl_answer)
            except Exception as e_sum:
                if is_quota_error(e_sum):
                    st.caption("Summary unavailable due to quota limits.")
                else:
                    st.caption("Summary unavailable.")
        with tabs[4]:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="results.csv", mime="text/csv")
            try:
                parquet_bytes = df.to_parquet(index=False)
                st.download_button("Download Parquet", data=parquet_bytes, file_name="results.parquet", mime="application/octet-stream")
            except Exception:
                pass

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
        except Exception as e_sum2:
            # Do not stop the app for summary errors
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

st.divider()
with st.expander("Schema and preview"):
    try:
        schema_df = get_schema_df(DATABASE_URL)
        st.subheader("yellow_trips schema")
        st.dataframe(schema_df, use_container_width=True)
        st.subheader("Latest rows preview")
        preview_df = preview_table(DATABASE_URL, limit=50)
        st.dataframe(preview_df, use_container_width=True)
    except Exception as _e:
        st.caption("Schema or preview unavailable (is the DB running and populated?).")
