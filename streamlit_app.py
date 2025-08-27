# Snowflake Streamlit App: Product Intelligence Dashboard with OpenAI Chatbot

import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.session import Session
import openai

# -------------------------------
# Create Snowpark Session Manually
# -------------------------------
connection_parameters = {
    "account": st.secrets["connections"]["snowflake"]["account"],
    "user": st.secrets["connections"]["snowflake"]["user"],
    "password": st.secrets["connections"]["snowflake"]["password"],
    "role": st.secrets["connections"]["snowflake"]["role"],
    "warehouse": st.secrets["connections"]["snowflake"]["warehouse"],
    "database": st.secrets["connections"]["snowflake"]["database"],
    "schema": st.secrets["connections"]["snowflake"]["schema"],
}

session = Session.builder.configs(connection_parameters).create()

# -------------------------------
# Load Data
# -------------------------------
df = session.table("reviews_with_sentiment").to_pandas()

# -------------------------------
# OpenAI API Key
# -------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------------
# App Title and Sidebar Filters
# -------------------------------
st.title("Product Intelligence Dashboard")

# Sidebar: Product filter
products = df['PRODUCT'].unique()
selected_products = st.sidebar.multiselect("Select Products:", options=products, default=products)

# Sidebar: Date filter
if "DATE" in df.columns:
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    min_date, max_date = df["DATE"].min(), df["DATE"].max()
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    start_date, end_date = None, None

# Filter dataframe
filtered_df = df[df['PRODUCT'].isin(selected_products)]
if start_date and end_date:
    filtered_df = filtered_df[(filtered_df["DATE"] >= pd.to_datetime(start_date)) & 
                              (filtered_df["DATE"] <= pd.to_datetime(end_date))]

# -------------------------------
# Data Preview
# -------------------------------
st.subheader("Data Preview")
st.dataframe(filtered_df.head())

# -------------------------------
# Average Sentiment by Region
# -------------------------------
st.subheader("Average Sentiment by Region")
if "REGION" in filtered_df.columns:
    region_sentiment = filtered_df.groupby("REGION")['SENTIMENT_SCORE'].mean().sort_values()
    chart_region = (
        alt.Chart(region_sentiment.reset_index())
        .mark_bar()
        .encode(
            x=alt.X('SENTIMENT_SCORE:Q', title="Avg Sentiment Score"),
            y=alt.Y('REGION:N', sort='-x', title="Region"),
            tooltip=['REGION', 'SENTIMENT_SCORE']
        )
        .properties(width=400, height=300)
    )
    st.altair_chart(chart_region, use_container_width=True)

# -------------------------------
# Delivery Issues by Region and Status
# -------------------------------
st.subheader("Sentiment Score by Region and Status for Each Product")
if all(col in filtered_df.columns for col in ["REGION","PRODUCT","STATUS","SENTIMENT_SCORE"]):
    grouped_issues = (
        filtered_df.groupby(['REGION', 'PRODUCT', 'STATUS'])['SENTIMENT_SCORE']
        .mean()
        .reset_index()
    )

    num_products = grouped_issues['PRODUCT'].nunique()
    num_cols = min(3, num_products)  # Max 3 columns per row

    base = alt.Chart(grouped_issues).mark_bar().encode(
        x=alt.X('REGION:N', title="Region"),
        y=alt.Y('SENTIMENT_SCORE:Q', title="Avg Sentiment Score"),
        color='STATUS:N',
        tooltip=['REGION', 'PRODUCT', 'STATUS', 'SENTIMENT_SCORE']
    )

    chart_faceted = base.facet(
        column=alt.Column('PRODUCT:N', title="Product", header=alt.Header(labelAngle=0))
    )

    st.altair_chart(chart_faceted, use_container_width=True)

# -------------------------------
# Chatbot Assistant Using OpenAI
# -------------------------------
st.subheader("Ask Questions About Your Data")
user_question = st.text_input("Enter your question here:")

if user_question:
    df_string = filtered_df.to_string(index=False)
    prompt = f"Answer this question using the dataset: {user_question} <context>{df_string}</context>"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    st.write(response['choices'][0]['message']['content'])
