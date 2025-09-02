"""
Configuration module for NexBuy
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for NexBuy recommendation system."""
    
    # Data configuration
    dataset_name: str = "vivek468/superstore-dataset-final"
    data_path: str = "data/"
    train_test_split_ratio: float = 0.8
    
    # Model configuration
    default_top_n: int = 5
    hybrid_weights_content: float = 0.4
    hybrid_weights_collab: float = 0.4
    hybrid_weights_popular: float = 0.2
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Streamlit configuration
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            dataset_name=os.getenv("DATASET_NAME", "vivek468/superstore-dataset-final"),
            data_path=os.getenv("DATA_PATH", "data/"),
            train_test_split_ratio=float(os.getenv("TRAIN_TEST_SPLIT_RATIO", "0.8")),
            default_top_n=int(os.getenv("DEFAULT_TOP_N", "5")),
            hybrid_weights_content=float(os.getenv("HYBRID_WEIGHTS_CONTENT", "0.4")),
            hybrid_weights_collab=float(os.getenv("HYBRID_WEIGHTS_COLLAB", "0.4")),
            hybrid_weights_popular=float(os.getenv("HYBRID_WEIGHTS_POPULAR", "0.2")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
            streamlit_host=os.getenv("STREAMLIT_HOST", "localhost"),
        )
    
    def validate(self):
        """Validate configuration values."""
        if self.train_test_split_ratio <= 0 or self.train_test_split_ratio >= 1:
            raise ValueError("train_test_split_ratio must be between 0 and 1")
        
        if self.default_top_n <= 0:
            raise ValueError("default_top_n must be positive")
        
        # Check that hybrid weights sum to 1.0 (approximately)
        total_weight = self.hybrid_weights_content + self.hybrid_weights_collab + self.hybrid_weights_popular
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Hybrid weights must sum to 1.0, got {total_weight}")
        
        return True


# Global configuration instance
config = Config.from_env()

__all__ = ['Config', 'config']
