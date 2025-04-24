import kagglehub
import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Download latest version
print("Dowlaod Dataset...")
path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
print(f"Dataset downloaded to: {path}")
csv_file_path = os.path.join(path, "Sample - Superstore.csv")
print(f"Reading data from: {csv_file_path}")

try:
    superstore_data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: File not found at {csv_file_path}.")
    exit()


# Keep necessary columns
columns_to_keep = ['Order ID', 'Order Date', 'Ship Date', 'Customer ID', 'Product ID', 'Product Name', 'Sales', 'Quantity', 'Category', 'Sub-Category']
superstore_data = superstore_data[columns_to_keep]

# Display the first 5 rows to check the data
print("First 5 rows of data:")
print(superstore_data.head())

# Convert dates
superstore_data['Order Date'] = pd.to_datetime(superstore_data['Order Date'])
superstore_data['Ship Date'] = pd.to_datetime(superstore_data['Ship Date'])

print(superstore_data[columns_to_keep].isnull().sum())

superstore_data.dropna(subset=columns_to_keep, inplace=True)

print("\n--- Finding Customers with Most Transactions ---")

# Count the number of rows (transaction line items) for each Customer ID
customer_transaction_counts = superstore_data.groupby('Customer ID').size()

# Sort the counts in descending order
customer_transaction_counts_sorted = customer_transaction_counts.sort_values(ascending=False)

print("Top 5 Customers by Number of Transaction Entries:")
print(customer_transaction_counts_sorted.head(5))

# Get the Customer ID with the absolute highest count
if not customer_transaction_counts_sorted.empty:
    top_customer_id = customer_transaction_counts_sorted.index[0]
    top_customer_count = customer_transaction_counts_sorted.iloc[0]
    print(f"\nCustomer with the most transaction entries: '{top_customer_id}' ({top_customer_count} entries)")
else:
    top_customer_id = None # Handle case where data might be empty
    print("\nCould not determine top customer.")
    

print("\n--- Splitting Data for Evaluation ---")
# Sort data by order date
superstore_data_sorted = superstore_data.sort_values('Order Date').copy()
superstore_data_sorted.reset_index(drop=True, inplace=True) # Optional: Reset index

# Define split point (e.g., 80% train, 20% test based on row count after sorting)
# Alternatively, pick a specific date for splitting
split_index = int(len(superstore_data_sorted) * 0.8)
train_df = superstore_data_sorted.iloc[:split_index].copy()
test_df = superstore_data_sorted.iloc[split_index:].copy()

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")
if not train_df.empty:
    print(f"Training data period: {train_df['Order Date'].min()} to {train_df['Order Date'].max()}")
if not test_df.empty:
    print(f"Testing data period: {test_df['Order Date'].min()} to {test_df['Order Date'].max()}")

# Identify users present in the test set for evaluation
test_users = test_df['Customer ID'].unique()
print(f"Number of unique users in test set: {len(test_users)}")