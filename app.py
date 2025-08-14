from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import networkx as nx
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Define input data model
class TransactionInput(BaseModel):
    wallet_address: str
    timestamp: str
    token_pair: str
    amount: float
    total_value_locked_usd: float
    liquidity_change: float
    price_usd: float
    price_change: float
    gasPrice: float
    gasUsed: float
    value: float
    from_address: str
    to_address: str

# Load pre-trained models and selected features
try:
    with open('xgb_binary.pkl', 'rb') as f:
        xgb_binary = pickle.load(f)
    with open('xgb_multi.pkl', 'rb') as f:
        xgb_multi = pickle.load(f)
    with open('iso_forest.pkl', 'rb') as f:
        iso_forest = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder_token.pkl', 'rb') as f:
        le_token = pickle.load(f)
    with open('label_encoder_fraud.pkl', 'rb') as f:
        le_fraud = pickle.load(f)
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    logger.info(f"Models and preprocessors loaded successfully. Token pairs in LabelEncoder: {list(le_token.classes_)}")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Predefined features (must match training)
features = [
    'log_amount', 'log_liquidity_change', 'amount_liquidity_interaction', 'amount_rolling_mean',
    'transaction_freq', 'total_volume', 'volume_change_rate', 'price_change',
    'gas_price_deviation', 'degree_centrality', 'token_pair_encoded',
    'price_volatility', 'tx_cluster', 'wallet_age_days', 'token_transfer_freq'
]

def compute_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute network-based features using transaction graph."""
    try:
        G = nx.DiGraph()
        for _, row in df.iterrows():
            if pd.notnull(row['from_address']) and pd.notnull(row['to_address']):
                G.add_edge(row['from_address'], row['to_address'], weight=row['value'] if pd.notnull(row['value']) else 0)
        degree_centrality = nx.degree_centrality(G)
        df['degree_centrality'] = df['wallet_address'].map(degree_centrality).fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error computing graph features: {str(e)}")
        return df

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input transaction data to match training feature set."""
    try:
        # Initial missing value handling
        df['amount'] = df['amount'].fillna(0)
        df['liquidity_change'] = df['liquidity_change'].fillna(0)
        df['total_value_locked_usd'] = df['total_value_locked_usd'].fillna(df['total_value_locked_usd'].mean())
        df['price_usd'] = df['price_usd'].fillna(df['price_usd'].mean())
        df['price_change'] = df['price_change'].fillna(0)
        df['gasPrice'] = df['gasPrice'].fillna(df['gasPrice'].mean())
        df['gasUsed'] = df['gasUsed'].fillna(df['gasUsed'].mean())
        df['value'] = df['value'].fillna(0) / 1e18

        # Transform to training features
        df['log_amount'] = np.log1p(df['amount'].clip(lower=0))
        df['log_liquidity_change'] = np.log1p(df['liquidity_change'].clip(lower=0))
        df['amount_liquidity_interaction'] = df['amount'] * df['liquidity_change']
        df['amount_rolling_mean'] = df['amount']  # Simplified for single transaction

        # Compute derived features (simplified for single transaction)
        df['tx_count'] = 1
        df['transaction_freq'] = 1
        df['total_volume'] = df['amount']
        df['volume_change_rate'] = 0
        gas_price_std = df['gasPrice'].std()
        df['gas_price_deviation'] = 0 if gas_price_std == 0 or pd.isna(gas_price_std) else (df['gasPrice'] - df['gasPrice'].mean()) / gas_price_std
        df['price_volatility'] = 0
        df['tx_cluster'] = 1
        df['wallet_age_days'] = 0
        df['token_transfer_freq'] = 1

        # Compute graph features
        df = compute_graph_features(df)

        # Encode categorical variables with handling for unseen labels
        def safe_label_encode(token_pair):
            if 'unknown' not in le_token.classes_:
                logger.error("LabelEncoder does not contain 'unknown'. Please retrain the model.")
                raise ValueError("LabelEncoder does not contain 'unknown'. Please retrain the model.")
            try:
                return le_token.transform([token_pair])[0]
            except ValueError:
                logger.warning(f"Unseen token pair: {token_pair}. Encoding as 'unknown'.")
                return le_token.transform(['unknown'])[0]

        df['token_pair_encoded'] = df['token_pair'].apply(safe_label_encode)

        # Select only the features used during training
        df = df[features]

        # Impute and scale features to match training
        imputer = SimpleImputer(strategy='mean')
        df[features] = imputer.fit_transform(df[features])
        df[features] = scaler.transform(df[features])

        # Ensure no NaN values
        if df[features].isna().any().any():
            logger.warning("NaN values detected after preprocessing. Filling with 0.")
            df[features] = df[features].fillna(0)

        return df
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise

@app.post("/predict_fraud")
async def predict_fraud(transaction: TransactionInput):
    """Endpoint to predict fraud for a single transaction."""
    try:
        # Convert input to DataFrame
        data = {
            'wallet_address': transaction.wallet_address,
            'timestamp': [datetime.strptime(transaction.timestamp, '%Y-%m-%d %H:%M:%S')],
            'token_pair': transaction.token_pair,
            'amount': transaction.amount,
            'total_value_locked_usd': transaction.total_value_locked_usd,
            'liquidity_change': transaction.liquidity_change,
            'price_usd': transaction.price_usd,
            'price_change': transaction.price_change,
            'gasPrice': transaction.gasPrice,
            'gasUsed': transaction.gasUsed,
            'value': transaction.value,
            'from_address': transaction.from_address,
            'to_address': transaction.to_address
        }
        df = pd.DataFrame(data)

        # Preprocess input
        df = preprocess_input(df)

        # Predict
        X = df[features]
        X_iso = df[selected_features]
        xgb_fraud_prob = xgb_binary.predict_proba(X)[:, 1][0]
        xgb_fraud_pred = xgb_binary.predict(X)[0]
        iso_fraud_pred = (iso_forest.predict(X_iso) == -1).astype(int)[0]
        is_fraud_pred = int(xgb_fraud_pred or iso_fraud_pred)
        fraud_type_pred = le_fraud.inverse_transform(xgb_multi.predict(X))[0]

        return {
            "wallet_address": transaction.wallet_address,
            "timestamp": transaction.timestamp,
            "token_pair": transaction.token_pair,
            "is_fraud_pred": bool(is_fraud_pred),
            "xgb_fraud_prob": float(xgb_fraud_prob),
            "fraud_type_pred": fraud_type_pred
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}