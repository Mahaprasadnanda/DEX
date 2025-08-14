import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import networkx as nx
import time
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_data(
    uniswap_file: str,
    wallet_tx_file: str,
    liquidity_file: str,
    price_file: str,
    fraud_file: str,
    real_time_tx_file: str
) -> tuple:
    """Load and merge all datasets."""
    logger.info("Loading datasets...")
    try:
        uniswap_df = pd.read_csv(uniswap_file)
        wallet_tx_df = pd.read_csv(wallet_tx_file)
        liquidity_df = pd.read_csv(liquidity_file)
        price_df = pd.read_csv(price_file)
        fraud_df = pd.read_csv(fraud_file)
        real_time_tx_df = pd.read_csv(real_time_tx_file)

        uniswap_df['timestamp'] = pd.to_datetime(uniswap_df['timestamp'])
        wallet_tx_df['timeStamp'] = pd.to_datetime(wallet_tx_df['timeStamp'], unit='s')
        liquidity_df['timestamp'] = pd.to_datetime(liquidity_df['timestamp'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        fraud_df['timestamp'] = pd.to_datetime(fraud_df['timestamp'])
        real_time_tx_df['timeStamp'] = pd.to_datetime(real_time_tx_df['timeStamp'])

        merged_df = uniswap_df.merge(
            real_time_tx_df,
            left_on=['wallet_address', 'timestamp'],
            right_on=['from', 'timeStamp'],
            how='left'
        )
        merged_df = merged_df.merge(
            liquidity_df,
            on=['token_pair', 'timestamp'],
            how='left'
        )
        merged_df['token'] = merged_df['token_pair'].str.split('/').str[0]
        merged_df = merged_df.merge(
            price_df,
            on=['token', 'timestamp'],
            how='left'
        )
        merged_df = merged_df.merge(
            fraud_df,
            on=['wallet_address', 'timestamp'],
            how='left'
        )
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df, wallet_tx_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def compute_graph_features(df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Compute network-based features using transaction graph."""
    logger.info("Computing graph features...")
    try:
        G = nx.DiGraph()
        for _, row in df.iterrows():
            if pd.notnull(row['from']) and pd.notnull(row['to']):
                G.add_edge(row['from'], row['to'], weight=row['value'] if pd.notnull(row['value']) else 0)
        for _, row in historical_df.iterrows():
            if pd.notnull(row['from']) and pd.notnull(row['to']):
                G.add_edge(row['from'], row['to'], weight=row['value'] if pd.notnull(row['value']) else 0)
        degree_centrality = nx.degree_centrality(G)
        pagerank = nx.pagerank(G, weight='weight')
        df['degree_centrality'] = df['wallet_address'].map(degree_centrality).fillna(0)
        df['pagerank'] = df['wallet_address'].map(pagerank).fillna(0)
        return df
    except Exception as e:
        logger.error(f"Error computing graph features: {str(e)}")
        return df

def preprocess_data(df: pd.DataFrame, historical_df: pd.DataFrame) -> tuple:
    """Preprocess data and compute features."""
    logger.info("Preprocessing data...")
    try:
        # Initial missing value handling
        df['amount'] = df['amount'].fillna(0)
        df['total_value_locked_usd'] = df['total_value_locked_usd'].fillna(df['total_value_locked_usd'].mean())
        df['liquidity_change'] = df['liquidity_change'].fillna(0)
        df['price_usd'] = df['price_usd'].fillna(df['price_usd'].mean())
        df['price_change'] = df['price_change'].fillna(0)
        df['gasPrice'] = df['gasPrice'].fillna(df['gasPrice'].mean())
        df['gasUsed'] = df['gasUsed'].fillna(df['gasUsed'].mean())
        df['is_fraud'] = df['is_fraud'].fillna(0)
        df['fraud_type'] = df['fraud_type'].fillna('normal')
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0) / 1e18

        # Feature engineering
        df['log_amount'] = np.log1p(df['amount'].clip(lower=0))
        df['log_liquidity_change'] = np.log1p(df['liquidity_change'].clip(lower=0))
        df['amount_liquidity_interaction'] = df['amount'] * df['liquidity_change']
        df['amount_rolling_mean'] = df.groupby('wallet_address')['amount'].transform(lambda x: x.rolling(window=5, min_periods=1).mean()).fillna(0)

        # Compute derived features
        df['tx_count'] = df.groupby('wallet_address')['timestamp'].transform('count')
        time_window = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
        df['transaction_freq'] = df['tx_count'] / max(time_window, 1)
        df['total_volume'] = df.groupby('wallet_address')['amount'].transform('sum')
        df['volume_change_rate'] = df.groupby('wallet_address')['amount'].pct_change().fillna(0)

        # Handle gas price deviation carefully
        gas_price_std = df['gasPrice'].std()
        if gas_price_std == 0 or pd.isna(gas_price_std):
            df['gas_price_deviation'] = 0
        else:
            df['gas_price_deviation'] = (df['gasPrice'] - df['gasPrice'].mean()) / gas_price_std

        # Add price volatility, transaction clustering, wallet age, and token transfer frequency
        df['price_volatility'] = df.groupby('token_pair')['price_usd'].transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)
        df['tx_cluster'] = df.groupby('wallet_address')['timestamp'].transform(lambda x: (x.diff().dt.total_seconds().fillna(3600) < 60).cumsum())
        df['wallet_age_days'] = (df['timestamp'].max() - df.groupby('wallet_address')['timestamp'].transform('min')).dt.total_seconds() / (24 * 3600)
        df['token_transfer_freq'] = df.groupby(['wallet_address', 'token_pair'])['timestamp'].transform('count') / max(time_window, 1)

        df = compute_graph_features(df, historical_df)

        le_token = LabelEncoder()
        # Explicitly add 'unknown' to token_pair to ensure it's in the encoder
        token_pairs = df['token_pair'].fillna('unknown').tolist() + ['unknown']
        df['token_pair_encoded'] = le_token.fit_transform(token_pairs)[:-1]  # Exclude the extra 'unknown'
        logger.info(f"Token pairs in LabelEncoder: {list(le_token.classes_)}")

        le_fraud = LabelEncoder()
        df['fraud_type_encoded'] = le_fraud.fit_transform(df['fraud_type'])

        # Log fraud type distribution
        logger.info(f"Fraud type distribution: {df['fraud_type'].value_counts()}")

        features = [
            'log_amount', 'log_liquidity_change', 'amount_liquidity_interaction', 'amount_rolling_mean',
            'transaction_freq', 'total_volume', 'volume_change_rate', 'price_change',
            'gas_price_deviation', 'degree_centrality', 'pagerank', 'token_pair_encoded',
            'price_volatility', 'tx_cluster', 'wallet_age_days', 'token_transfer_freq'
        ]

        # Correlation analysis to remove redundant features
        corr_matrix = df[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        features = [f for f in features if f not in to_drop]
        logger.info(f"Features after correlation analysis: {features}")

        # Impute NaN values
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(df[features])
        df[features] = imputed_data

        # Scale features
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

        # Check for NaN after scaling
        if df[features].isna().any().any():
            logger.warning("NaN values detected after scaling. Filling with 0.")
            df[features] = df[features].fillna(0)

        logger.info(f"Features computed: {features}")
        logger.info(f"Feature matrix shape: {df[features].shape}")
        return df, features, scaler, le_token, le_fraud
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def select_features(df: pd.DataFrame, xgb_binary, features: list) -> list:
    """Select top 10 features using permutation importance, ensuring critical features."""
    logger.info("Selecting features for IsolationForest...")
    try:
        xgb_binary.fit(df[features], df['is_fraud'])
        perm_importance = permutation_importance(xgb_binary, df[features], df['is_fraud'], n_repeats=10, random_state=42)
        feature_importance = pd.Series(perm_importance.importances_mean, index=features)
        logger.info(f"Feature importances:\n{feature_importance.sort_values(ascending=False)}")
        top_features = feature_importance.nlargest(10).index.tolist()
        must_have = ['log_amount', 'log_liquidity_change', 'amount_liquidity_interaction']
        top_features = list(set(top_features + must_have))[:10]
        logger.info(f"Selected features: {top_features}")
        return top_features
    except Exception as e:
        logger.error(f"Error selecting features: {str(e)}")
        return features

def evaluate_isolation_forest(iso_forest, X, y, n_splits=5):
    """Evaluate IsolationForest using cross-validation."""
    logger.info("Evaluating IsolationForest with cross-validation...")
    try:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            iso_forest.fit(X_train)
            y_pred = (iso_forest.predict(X_test) == -1).astype(int)
            f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
            scores.append(f1)
        logger.info(f"IsolationForest CV F1-score (fraud): {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
        return np.mean(scores)
    except Exception as e:
        logger.error(f"Error evaluating IsolationForest: {str(e)}")
        return 0

def train_models(df: pd.DataFrame, features: list) -> tuple:
    """Train supervised and unsupervised models."""
    logger.info("Training models...")
    try:
        X = df[features]
        y_binary = df['is_fraud'].fillna(0)
        y_multi = df['fraud_type_encoded'].fillna(0)

        # Check for NaN in X
        if X.isna().any().any():
            raise ValueError("Input X contains NaN after preprocessing.")

        # Log original shapes
        logger.info(f"Original X shape: {X.shape}, y_binary shape: {y_binary.shape}, y_multi shape: {y_multi.shape}")

        # Separate SMOTE for binary and multi-class
        smote_binary = SMOTE(random_state=42, k_neighbors=2)
        smote_multi = SMOTE(random_state=42, sampling_strategy={1: 200, 2: 200}, k_neighbors=2)
        X_binary_resampled, y_binary_resampled = smote_binary.fit_resample(X, y_binary)
        X_multi_resampled, y_multi_resampled = smote_multi.fit_resample(X, y_multi)

        # Log resampled shapes
        logger.info(f"Binary resampled X shape: {X_binary_resampled.shape}, y_binary shape: {y_binary_resampled.shape}")
        logger.info(f"Multi resampled X shape: {X_multi_resampled.shape}, y_multi shape: {y_multi_resampled.shape}")

        xgb_binary = XGBClassifier(
            random_state=42, n_jobs=-1, max_depth=5, reg_lambda=1.0, reg_alpha=0.1,
            scale_pos_weight=len(y_binary_resampled[y_binary_resampled==0])/len(y_binary_resampled[y_binary_resampled==1])
        )
        scores = cross_val_score(xgb_binary, X_binary_resampled, y_binary_resampled, cv=5, scoring='f1_weighted')
        logger.info(f"XGBoost Binary CV F1-score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        xgb_binary.fit(X_binary_resampled, y_binary_resampled)

        xgb_multi = XGBClassifier(random_state=42, n_jobs=-1, max_depth=5, reg_lambda=1.0, reg_alpha=0.1)
        scores = cross_val_score(xgb_multi, X_multi_resampled, y_multi_resampled, cv=5, scoring='f1_weighted')
        logger.info(f"XGBoost Multi-class CV F1-score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        xgb_multi.fit(X_multi_resampled, y_multi_resampled)

        selected_features = select_features(df, xgb_binary, features)
        iso_forest = IsolationForest(contamination=0.5, random_state=42, n_estimators=200, max_samples=512, n_jobs=-1)
        evaluate_isolation_forest(iso_forest, df[selected_features], y_binary)
        iso_forest.fit(df[selected_features])

        logger.info("Models trained successfully.")
        return xgb_binary, xgb_multi, iso_forest, selected_features
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def predict_fraud(df: pd.DataFrame, xgb_binary, xgb_multi, iso_forest, features: list, selected_features: list) -> pd.DataFrame:
    """Predict fraud using both models."""
    logger.info("Predicting fraud...")
    try:
        start_time = time.time()
        X = df[features]
        X_iso = df[selected_features]
        df['xgb_fraud_prob'] = xgb_binary.predict_proba(X)[:, 1]
        df['xgb_fraud_pred'] = xgb_binary.predict(X)
        df['xgb_fraud_type_pred'] = xgb_multi.predict(X)
        df['iso_fraud_pred'] = (iso_forest.predict(X_iso) == -1).astype(int)
        df['is_fraud_pred'] = (df['xgb_fraud_pred'] | df['iso_fraud_pred']).astype(int)

        le_fraud = LabelEncoder().fit(df['fraud_type'])
        df['fraud_type_pred'] = le_fraud.inverse_transform(df['xgb_fraud_type_pred'])

        latency = (time.time() - start_time) * 1000 / len(df)
        logger.info(f"Prediction latency: {latency:.2f} ms per transaction")
        return df
    except Exception as e:
        logger.error(f"Error predicting fraud: {str(e)}")
        raise

def main():
    """Main function to run the fraud detection pipeline."""
    try:
        df, wallet_tx_df = load_raw_data(
            'uniswap_transactions.csv',
            'wallet_tx.csv',
            'liquidity_pool_data.csv',
            'token_price_history.csv',
            'labeled_fraud_data.csv',
            'real_time_transactions.csv'
        )
        df, features, scaler, le_token, le_fraud = preprocess_data(df, wallet_tx_df)
        xgb_binary, xgb_multi, iso_forest, selected_features = train_models(df, features)
        df = predict_fraud(df, xgb_binary, xgb_multi, iso_forest, features, selected_features)

        # Save models and preprocessors
        with open('xgb_binary.pkl', 'wb') as f:
            pickle.dump(xgb_binary, f)
        with open('xgb_multi.pkl', 'wb') as f:
            pickle.dump(xgb_multi, f)
        with open('iso_forest.pkl', 'wb') as f:
            pickle.dump(iso_forest, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('label_encoder_token.pkl', 'wb') as f:
            pickle.dump(le_token, f)
        with open('label_encoder_fraud.pkl', 'wb') as f:
            pickle.dump(le_fraud, f)
        with open('selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)
        logger.info("Models and preprocessors saved to .pkl files")

        if 'is_fraud' in df.columns:
            logger.info("XGBoost Binary Classification Report:")
            print(classification_report(df['is_fraud'], df['xgb_fraud_pred']))
            logger.info("Isolation Forest Classification Report:")
            print(classification_report(df['is_fraud'], df['iso_fraud_pred']))
            logger.info("XGBoost Multi-class Classification Report:")
            print(classification_report(df['fraud_type_encoded'], df['xgb_fraud_type_pred']))

        output_df = df[['wallet_address', 'timestamp', 'token_pair', 'is_fraud_pred', 'xgb_fraud_prob', 'fraud_type', 'fraud_type_pred']]
        output_df.to_csv('fraud_predictions.csv', index=False)
        logger.info("Fraud predictions saved to fraud_predictions.csv")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()