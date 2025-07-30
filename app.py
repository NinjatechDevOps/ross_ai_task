import os
import streamlit as st
import pandas as pd
import sqlite3
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import difflib

# Load environment variables
load_dotenv()

# Streamlit UI for API key
st.sidebar.header("Configuration")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Determine which API key to use
if user_api_key:
    api_key = user_api_key
    st.sidebar.success("API Key provided!")
else:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.sidebar.warning("OpenAI API Key not found. Please provide one.")
        st.stop()

# Connect to SQLite DB
db = SQLDatabase.from_uri("sqlite:///products.db")

# Instantiate LLM with tools support
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=api_key)

# Add the SQL database toolkit as tools
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()

# --- Fuzzy Matching Tool ---
@tool
def fuzzy_product_search(product_name: str) -> str:
    """
    Suggests the closest product names in the database to the given (possibly misspelled) product_name.
    Uses Levenshtein distance (via difflib) to find matches with distance <= 2.
    Supports all product tables (laptops, phones, etc).
    """
    conn = sqlite3.connect('products.db')
    try:
        # Find all tables that have a 'name' column
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        all_names = []
        for table in tables['name']:
            # Check if table has a 'name' column
            try:
                cols = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                if 'name' in cols['name'].values:
                    # Try to get both brand and name if brand exists
                    if 'brand' in cols['name'].values:
                        df = pd.read_sql_query(f"SELECT DISTINCT brand, name FROM {table}", conn)
                        # Combine brand and name for fuzzy matching
                        combined = [f"{row['brand']} {row['name']}" for _, row in df.iterrows()]
                        all_names.extend(combined)
                        # Also include just the name for fallback
                        all_names.extend(df['name'].tolist())
                    else:
                        df = pd.read_sql_query(f"SELECT DISTINCT name FROM {table}", conn)
                        all_names.extend(df['name'].tolist())
            except Exception:
                continue
        all_names = list(set(all_names))
        # print(f"all_names---{all_names}")
        # print(f"product_name---{product_name}")
        close_matches = difflib.get_close_matches(product_name, all_names, n=7, cutoff=0.4)
        print(f"closest match ----{close_matches}")
        if not close_matches:
            # Try more lenient cutoff
            close_matches = [n for n in all_names if difflib.SequenceMatcher(None, product_name.lower(), n.lower()).ratio() > 0.6]
        if close_matches:
            return f"Did you mean: {', '.join(close_matches)}?"
        else:
            return "No similar product names found."
    finally:
        conn.close()

# --- Variant Search Tool ---
@tool
def variant_search(product_name: str) -> str:
    """
    Returns all variants for a given product name, searching across all product tables.
    """
    conn = sqlite3.connect('products.db')
    try:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        found = False
        results = []
        for table in tables['name']:
            # Check if table has a 'name' column
            try:
                cols = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                if 'name' in cols['name'].values:
                    df = pd.read_sql_query(f"SELECT * FROM {table} WHERE name LIKE ?", conn, params=(f"%{product_name}%",))
                    if not df.empty:
                        found = True
                        # Try to show variant/price/specs if present, else show all columns
                        for _, row in df.iterrows():
                            variant = row['variant'] if 'variant' in df.columns else ""
                            price = row['price'] if 'price' in df.columns else (row['price_usd'] if 'price_usd' in df.columns else "")
                            specs = row['specs'] if 'specs' in df.columns else ""
                            # Fallback: show all columns if not standard
                            if not variant and not price and not specs:
                                row_str = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                                results.append(f"{table}: {row_str}")
                            else:
                                results.append(f"{row['name']} ({variant}) - ${price}, {specs}".strip(" ,"))
            except Exception:
                continue
        if not found:
            return "No variants found for this product."
        else:
            return "\n".join(results)
    finally:
        conn.close()

# Combine tools, including fuzzy and variant search
tools = sql_tools + [fuzzy_product_search, variant_search]

# Set up memory to maintain conversation
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# --- Prompt Templates for Scenario Adaptation ---
prompt_search = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful product advisor. Answer the user's product search queries using the database. If no product is found, suggest similar names using the fuzzy_product_search tool."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

prompt_compare = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful product advisor. The user wants to compare two or more products. Present a clear comparison. If a product is not found, suggest similar names using the fuzzy_product_search tool."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

prompt_recommend = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful product advisor. Recommend a product based on the user's preferences and context. If the user mentions a product that is not found, suggest similar names using the fuzzy_product_search tool."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Simple scenario detection for prompt adaptation
def detect_scenario(user_input: str):
    if "compare" in user_input.lower():
        return "compare"
    elif "recommend" in user_input.lower() or "suggest" in user_input.lower():
        return "recommend"
    else:
        return "search"

# Create the agent executor with dynamic prompt selection
def get_agent_executor(scenario):
    if scenario == "compare":
        prompt = prompt_compare
    elif scenario == "recommend":
        prompt = prompt_recommend
    else:
        prompt = prompt_search
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        verbose=True
    )

# Function to fetch data from the database
def get_db_data():
    conn = sqlite3.connect('products.db')
    try:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        data = {}
        for table_name in tables['name']:
            data[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return data
    finally:
        conn.close()

# Streamlit UI
st.set_page_config(page_title="Product Advisor Chatbot", layout="wide")

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .st-emotion-cache-1y4p8pa {
        width: 100%;
        padding: 1rem;
        border-top: 1px solid #ddd;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Chatbot", "View Product Data"])

with tab1:
    st.header("Chat with the Advisor")

    # Chat container
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.memory.chat_memory.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

    # Fixed input bar at the bottom
    with st.container():
        user_input = st.chat_input("Ask about products...")
        if user_input:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

                try:
                    with st.spinner("Thinking..."):
                        scenario = detect_scenario(user_input)
                        agent_executor = get_agent_executor(scenario)
                        response = agent_executor.invoke({"input": user_input})
                    with st.chat_message("assistant"):
                        st.markdown(response["output"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with tab2:
    st.header("Product Database")
    try:
        db_data = get_db_data()
        if db_data:
            for table_name, df in db_data.items():
                st.subheader(f"Table: {table_name}")
                st.dataframe(df)
        else:
            st.write("No tables found in the database.")
    except Exception as e:
        st.error(f"Could not load data from database: {e}")