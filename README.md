# NexBuy 🛒

**NexBuy** is a multi-strategy product recommendation system that intelligently connects customers with the right products using state-of-the-art machine learning techniques.

## 🎯 Features

- **Multi-Strategy Approach**: Combines Popular, Content-Based, Collaborative, and Hybrid recommendation algorithms
- **Real-time Evaluation**: Built-in precision metrics and live evaluation dashboard
- **Scalable Architecture**: Modular design for easy extension and maintenance
- **Interactive Web App**: Streamlit-based interface for easy testing and demonstration
- **Production Ready**: Comprehensive testing, logging, and configuration management

### Recommendation Strategies

1. **Popular-Based Filtering**: Recommends globally popular products based on sales/quantity metrics
2. **Content-Based Filtering**: Uses TF-IDF and cosine similarity on product features (name, category, sub-category)
3. **Collaborative Filtering**: Item-item collaborative filtering based on user purchase patterns
4. **Hybrid Approach**: Combines all three methods with configurable weights

### Data Pipeline

1. **Data Loading**: Automatic download from Kaggle (Superstore dataset)
2. **Preprocessing**: Data cleaning, feature engineering, and train/test splitting
3. **Precomputations**: TF-IDF matrices, similarity matrices, and popularity metrics
4. **Model Training**: Fit recommendation models on training data
5. **Evaluation**: Precision@K metrics on test data

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nancy-kataria/NexBuy.git
   cd NexBuy
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate on macOS/Linux
   source .venv/bin/activate
   
   # Activate on Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   ```bash
   cp .env.template .env
   # Edit .env with your specific configurations
   ```

### 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`

## 📊 Project Architecture

### Project Structure

```
NexBuy/
├── src/
│   └── nexbuy/
│       ├── data/           # Data preprocessing modules
│       ├── models/         # Recommendation algorithms
│       ├── utils/          # Helper functions and calculations
│       └── evaluation/     # Model evaluation metrics
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for analysis
├── config/                 # Configuration files
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # Project documentation
```


[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Scikit-learn]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[Pandas]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[NumPy]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[Kaggle]: https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white
[Streamlit]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white


[scikit-learn-url]: https://scikit-learn.org/stable/
[python-url]: https://www.python.org/
[streamlit-url]: https://streamlit.io/
[numpy-url]: https://numpy.org/
[kaggle-url]: https://www.kaggle.com/
[pandas-url]: https://pandas.pydata.org/
