import kagglehub
import pandas as pd
import os
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing for the recommendation system."""
    
    def __init__(self, dataset_name: str = "vivek468/superstore-dataset-final", verbose: bool = True):
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.superstore_data = None
        self.train_df = None
        self.test_df = None
        
        # Configure pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
    
    def download_and_load_data(self) -> pd.DataFrame:
        """Download and load the dataset."""
        if self.verbose:
            print("Download Dataset...")
            
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            if self.verbose:
                print(f"Dataset downloaded to: {path}")
                
            csv_file_path = os.path.join(path, "Sample - Superstore.csv")
            if self.verbose:
                print(f"Reading data from: {csv_file_path}")
            
            self.superstore_data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            if self.verbose:
                print("Data loaded successfully.")
                
        except FileNotFoundError:
            logger.error(f"File not found at {csv_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
        return self.superstore_data
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the data."""
        if self.superstore_data is None:
            raise ValueError("Data must be loaded first using download_and_load_data()")
        
        # Keep necessary columns
        columns_to_keep = [
            'Order ID', 'Order Date', 'Ship Date', 'Customer ID', 
            'Product ID', 'Product Name', 'Sales', 'Quantity', 
            'Category', 'Sub-Category'
        ]
        
        self.superstore_data = self.superstore_data[columns_to_keep]
        
        # Optionally show first 5 rows (commented out to reduce CLI noise)
        # if self.verbose:
        #     print("First 5 rows of data:")
        #     print(self.superstore_data.head())
        
        # Convert dates
        self.superstore_data['Order Date'] = pd.to_datetime(self.superstore_data['Order Date'])
        self.superstore_data['Ship Date'] = pd.to_datetime(self.superstore_data['Ship Date'])
        
        # Check for null values
        if self.verbose:
            print("Null values per column:")
            print(self.superstore_data[columns_to_keep].isnull().sum())
        
        # Drop rows with null values
        self.superstore_data.dropna(subset=columns_to_keep, inplace=True)
        
        return self.superstore_data
    
    def analyze_customers(self) -> None:
        """Analyze customer transaction patterns."""
        if self.superstore_data is None:
            raise ValueError("Data must be cleaned first")
            
        if self.verbose:
            print("\n--- Finding Customers with Most Transactions ---")
        
        # Count transactions per customer
        customer_transaction_counts = self.superstore_data.groupby('Customer ID').size()
        customer_transaction_counts_sorted = customer_transaction_counts.sort_values(ascending=False)
        
        if self.verbose:
            print("Top 5 Customers by Number of Transaction Entries:")
            print(customer_transaction_counts_sorted.head(5))
        
        # Get top customer
        if not customer_transaction_counts_sorted.empty:
            top_customer_id = customer_transaction_counts_sorted.index[0]
            top_customer_count = customer_transaction_counts_sorted.iloc[0]
            if self.verbose:
                print(f"\nCustomer with most transactions: '{top_customer_id}' ({top_customer_count} entries)")
        else:
            logger.warning("Could not determine top customer")
    
    def split_data(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets chronologically."""
        if self.superstore_data is None:
            raise ValueError("Data must be processed first")
            
        if self.verbose:
            print("\n--- Splitting Data for Evaluation ---")
        
        # Sort by order date
        superstore_data_sorted = self.superstore_data.sort_values('Order Date').copy()
        superstore_data_sorted.reset_index(drop=True, inplace=True)
        
        # Split data
        split_index = int(len(superstore_data_sorted) * train_ratio)
        self.train_df = superstore_data_sorted.iloc[:split_index].copy()
        self.test_df = superstore_data_sorted.iloc[split_index:].copy()
        
        if self.verbose:
            print(f"Training data shape: {self.train_df.shape}")
            print(f"Testing data shape: {self.test_df.shape}")
            
            if not self.train_df.empty:
                print(f"Training data period: {self.train_df['Order Date'].min()} to {self.train_df['Order Date'].max()}")
            if not self.test_df.empty:
                print(f"Testing data period: {self.test_df['Order Date'].min()} to {self.test_df['Order Date'].max()}")
            
            test_users = self.test_df['Customer ID'].unique()
            print(f"Number of unique users in test set: {len(test_users)}")
        
        return self.train_df, self.test_df
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get all processed data (full dataset, train, test)."""
        if self.superstore_data is None or self.train_df is None or self.test_df is None:
            raise ValueError("Data must be fully processed first")
        return self.superstore_data, self.train_df, self.test_df


# Initialize preprocessor and process data
preprocessor = DataPreprocessor()
superstore_data = preprocessor.download_and_load_data()
superstore_data = preprocessor.clean_data()
preprocessor.analyze_customers()
train_df, test_df = preprocessor.split_data()

# Export for backward compatibility
__all__ = ['preprocessor', 'superstore_data', 'train_df', 'test_df', 'DataPreprocessor']
