import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# Generate summary statistics for cost-related columns
cost_summary = data[cost_columns].describe()
print(cost_summary)

# Step 5: Visualize the Data
# Use a boxplot to visualize the distribution of costs across different pricing options.



# Set up the plotting style
sns.set_theme(style="whitegrid")

# Create a boxplot to visualize the distribution of costs
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette="Set2")

# Set the plot labels and title
plt.title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
#Detect potential outliers in the cost data using interquartile ranges (IQR).
# Function to identify outliers using the IQR method
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Find outliers in the On-Demand cost column
outliers_on_demand = detect_outliers('On Demand')
print(outliers_on_demand)

# Compare Reserved and On-Demand costs
cost_comparison = (
    data[['Name', 'On Demand', 'Linux Reserved cost']]
    .dropna()
    .sort_values('On Demand')
)

print(cost_comparison.head(10))  # Display the top 10 cost-effective instances
# 1. Filter for specific instance families

def filter_instance_family(family):
    return data[data['Name'].str.startswith(family)]


# Filter for T2 and T3 instance families
t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')


# Compare summary statistics for T2 and T3 instances
t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

print("T2 Instance Costs Summary:\n", t2_summary)
print("\nT3 Instance Costs Summary:\n", t3_summary)


# 2. Visualize the cost distribution of different instance families

# T2 instances
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=t2_instances[cost_columns],
    palette="Blues",
    showmeans=True
)

plt.title('Cost Distribution for T2 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.show()


# T3 instances
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=t3_instances[cost_columns],
    palette="Greens",
    showmeans=True
)

plt.title('Cost Distribution for T3 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.show()


# 3. Compare On-Demand and Reserved costs for different instance families

comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])

# Sort by On-Demand costs
comparison_sorted = (
    comparison
    .dropna()
    .sort_values('On Demand')
)

print(comparison_sorted.head(10))
