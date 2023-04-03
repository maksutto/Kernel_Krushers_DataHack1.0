import streamlit as st
import snowflake.connector as snowflake
import pandas as pd

# Snowflake connection parameters
account = 'https://xclcdmf-ob08539.snowflakecomputing.com'
user = 'DEVPATEL'
password = ''
database = 'STOCKS'
schema = 'BANK'
warehouse = 'COMPUTE_WH'

# Establish Snowflake connection
conn = snowflake.connect(
    account=account,
    user=user,
    password=password,
    database=database,
    schema=schema,
    warehouse=warehouse
)

# SQL query to retrieve training data from a table
train_query = 'SELECT * FROM Bank'

# Execute SQL query and retrieve training data
train_df = pd.read_sql(train_query, conn)

# Streamlit app


# Close Snowflake connection
conn.close()