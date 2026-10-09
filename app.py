import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="EC2 Cost Analysis", layout="wide")
st.title("Analyzing Amazon EC2 Instance Costs")

# Load data
file_path = 'ec2dataset.csv'
data = pd.read_csv(file_path)
cost_columns = [
    'On Demand',
    'Linux Reserved cost',
    'Linux Spot Minimum cost',
    'Windows On Demand cost',
    'Windows Reserved cost'
]

# Remove '$' and 'hourly' and convert columns to numeric
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

# Summary statistics
st.header("1. Summary Statistics of Cost Columns")
cost_summary = data[cost_columns].describe()
st.dataframe(cost_summary)

# Step 5: Visualize the Data
st.header("2. Cost Comparison (Boxplot)")
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette="Set2", ax=ax)
ax.set_title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
ax.set_ylabel('Cost (USD)', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
fig.tight_layout()
st.pyplot(fig)

# Detect potential outliers using IQR
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

st.header("3. Outliers in On-Demand Cost (IQR Method)")
outliers_on_demand = detect_outliers('On Demand')
st.write(f"Number of outliers: {len(outliers_on_demand)}")
st.dataframe(outliers_on_demand)

# Compare Reserved and On-Demand costs
st.header("4. Top 10 Cost-Effective Instances (On-Demand vs Reserved)")
cost_comparison = (
    data[['Name', 'On Demand', 'Linux Reserved cost']]
    .dropna()
    .sort_values('On Demand')
)
st.dataframe(cost_comparison.head(10))

# Filter for specific instance families
def filter_instance_family(family):
    return data[data['Name'].str.startswith(family)]

t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')

st.header("5. T2 vs T3 Instance Families")
col1, col2 = st.columns(2)
with col1:
    st.subheader("T2 Instance Costs Summary")
    st.dataframe(t2_instances[cost_columns].describe())
with col2:
    st.subheader("T3 Instance Costs Summary")
    st.dataframe(t3_instances[cost_columns].describe())

# T2 boxplot
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=t2_instances[cost_columns], palette="Blues", showmeans=True, ax=ax)
ax.set_title('Cost Distribution for T2 Instances', fontsize=16)
ax.set_ylabel('Cost (USD)', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
fig.tight_layout()
st.pyplot(fig)

# T3 boxplot
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=t3_instances[cost_columns], palette="Greens", showmeans=True, ax=ax)
ax.set_title('Cost Distribution for T3 Instances', fontsize=16)
ax.set_ylabel('Cost (USD)', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
fig.tight_layout()
st.pyplot(fig)

# Compare On-Demand and Reserved costs for T2 and T3
st.header("6. T2 & T3: On-Demand vs Reserved (Top 10 Cheapest)")
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])
comparison_sorted = comparison.dropna().sort_values('On Demand')
st.dataframe(comparison_sorted.head(10))
