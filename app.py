import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="EC2 Cost Analysis", layout="wide")
st.title("Analyzing Amazon EC2 Instance Costs")
st.caption("Part 1: Cost analysis & outliers  |  Part 2: Predicting On-Demand cost with Linear Regression")

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

st.header("Part 1: Cost Analysis")
st.subheader("Dataset Preview")
st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
st.dataframe(data.head())
st.subheader("Missing Values After Cleaning")
st.dataframe(data[cost_columns].isnull().sum().rename("Missing values"))

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


# ==========================================================
# Part 2: Predicting EC2 Instance Costs Using Regression
# ==========================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.header("Part 2: Predicting EC2 Instance Costs Using Regression Analysis")

# Step 3: Feature Engineering
reg_data = data.copy()
reg_data['Instance Memory'] = pd.to_numeric(
    reg_data['Instance Memory'].str.replace(' GiB', '').str.replace(',', ''),
    errors='coerce'
)
reg_data['vCPUs'] = pd.to_numeric(reg_data['vCPUs'].str.extract(r'(\d+)', expand=False))

st.subheader("Feature Engineering (Memory & vCPUs as numbers)")
st.dataframe(reg_data[['Name', 'Instance Memory', 'vCPUs', 'On Demand']].head())

# Step 4: Handle Missing Data
data_no_missing = reg_data.dropna(subset=['On Demand', 'Instance Memory', 'vCPUs'])
st.write(f"Rows after dropping missing values: {len(data_no_missing)}")

# Remove On-Demand cost outliers (same IQR method as Part 1).
# A few huge instances (e.g. U7IN-32TB at $407/hr) pull the regression line
# so much that the intercept becomes negative and small instances get
# negative predicted costs. Removing them gives a realistic model.
Q1 = data_no_missing['On Demand'].quantile(0.25)
Q3 = data_no_missing['On Demand'].quantile(0.75)
IQR = Q3 - Q1
data_cleaned = data_no_missing[
    (data_no_missing['On Demand'] >= Q1 - 1.5 * IQR) &
    (data_no_missing['On Demand'] <= Q3 + 1.5 * IQR)
]
st.write(f"Outliers removed (IQR method): {len(data_no_missing) - len(data_cleaned)}  |  "
         f"Rows used for regression: {len(data_cleaned)}")
st.info("Extreme high-cost instances (outliers) are removed before training. "
        "With them included, the intercept becomes negative and small instances "
        "get negative predicted costs, which is not realistic.")

# Step 5: Split the Data
X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Step 6: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Model Coefficients")
c1, c2, c3 = st.columns(3)
c1.metric("Intercept", f"{model.intercept_:.4f}")
c2.metric("Memory coefficient (per GiB)", f"{model.coef_[0]:.4f}")
c3.metric("vCPU coefficient (per vCPU)", f"{model.coef_[1]:.4f}")

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

st.subheader("Model Evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.4f}")
m2.metric("MSE", f"{mse:.4f}")
m3.metric("RMSE", f"{rmse:.4f}")

# Step 8: Visualize the Results
st.subheader("Actual vs Predicted On-Demand Costs")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.7, color='b')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_title('Actual vs Predicted On-Demand Costs')
ax.set_xlabel('Actual On-Demand Cost')
ax.set_ylabel('Predicted On-Demand Cost')
st.pyplot(fig)

# Step 9: Make Predictions
st.subheader("Predict Cost for a New Instance")
p1, p2 = st.columns(2)
memory_in = p1.number_input("Instance Memory (GiB)", min_value=0.5, value=4.0, step=0.5)
vcpu_in = p2.number_input("vCPUs", min_value=1, value=2, step=1)
new_instance = pd.DataFrame([[memory_in, vcpu_in]], columns=['Instance Memory', 'vCPUs'])
predicted_cost = model.predict(new_instance)
st.success(f"Predicted On-Demand Cost for {memory_in} GiB, {vcpu_in} vCPUs: ${predicted_cost[0]:.4f} per hour")
