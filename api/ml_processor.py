import pandas as pd
import numpy as np
import json
import logging
import os
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("scikit-learn not available. ML features will be disabled.")

# Advanced ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("XGBoost/LightGBM not available. Advanced ML features will be disabled.")

# AI/LLM Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. LLM features will be disabled.")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available. Chart features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLProcessor:
    """
    Advanced AI-powered Machine Learning processor for Excel data analysis and prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.ml_available = ML_AVAILABLE
        self.advanced_ml_available = ADVANCED_ML_AVAILABLE
        self.openai_available = OPENAI_AVAILABLE
        self.visualization_available = VISUALIZATION_AVAILABLE
        
        # Initialize OpenAI if available
        if self.openai_available:
            self.openai_client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        
        # Model registry for different tasks
        self.model_registry = {
            'classification': {
                'simple': [LogisticRegression, DecisionTreeClassifier, GaussianNB],
                'ensemble': [RandomForestClassifier, GradientBoostingClassifier],
                'advanced': [SVC, MLPClassifier, KNeighborsClassifier]
            },
            'regression': {
                'simple': [LinearRegression, Ridge, Lasso],
                'ensemble': [RandomForestRegressor, GradientBoostingRegressor],
                'advanced': [SVR, MLPRegressor, KNeighborsRegressor]
            },
            'clustering': {
                'simple': [KMeans],
                'advanced': [DBSCAN, AgglomerativeClustering]
            }
        }
        
        if not self.ml_available:
            logger.warning("Machine learning features are not available. Please install scikit-learn.")
    
    def predict_values(self, df: pd.DataFrame, target_column: str, feature_columns: List[str], 
                      model_type: str = 'auto', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a model and make predictions with advanced AI capabilities
        """
        if not self.ml_available:
            return {'error': 'Machine learning not available. Please install scikit-learn.'}
        
        try:
            # Prepare data
            if target_column not in df.columns:
                return {'error': f'Target column "{target_column}" not found in data'}
            
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                return {'error': f'Feature columns not found: {missing_features}'}
            
            # Prepare features and target
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
            y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else y.mode()[0])
            
            # Encode categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[f'{col}_encoder'] = le
            
            # Determine if classification or regression
            is_classification = not pd.api.types.is_numeric_dtype(y)
            if is_classification:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                self.encoders[f'{target_column}_encoder'] = le_target
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[f'{target_column}_scaler'] = scaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y if is_classification else None
            )
            
            # Auto-select best model
            if model_type == 'auto':
                best_model, best_score = self._auto_select_model(X_train, y_train, is_classification)
            else:
                best_model = self._get_model_by_type(model_type, is_classification)
                best_model.fit(X_train, y_train)
                best_score = best_model.score(X_test, y_test)
            
            # Train final model
            best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2_score': r2,
                    'mae': np.mean(np.abs(y_test - y_pred))
                }
            
            # Feature importance
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                importance_scores = best_model.feature_importances_
                feature_importance = dict(zip(feature_columns, importance_scores))
            elif hasattr(best_model, 'coef_'):
                coef_scores = np.abs(best_model.coef_[0] if len(best_model.coef_.shape) > 1 else best_model.coef_)
                feature_importance = dict(zip(feature_columns, coef_scores))
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
            
            model_key = f"{target_column}_{model_type}"
            self.models[model_key] = best_model

            insights = self._generate_model_insights(
                df, target_column, feature_columns, metrics, feature_importance, is_classification
            )

            return {
                'model_type': type(best_model).__name__,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'is_classification': is_classification,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'cross_val_scores': cv_scores.tolist(),
                'cross_val_mean': cv_scores.mean(),
                'cross_val_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'insights': insights,
                'model_performance': {
                    'train_score': best_model.score(X_train, y_train),
                    'test_score': best_model.score(X_test, y_test),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f'Feature columns not found: {missing}')

        features = df[feature_columns].copy()
        for col in features.select_dtypes(include=['object']).columns:
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.fillna(features.mean(numeric_only=True))
        return features

    def cluster_data(self, df: pd.DataFrame, feature_columns: List[str],
                    n_clusters: int = 3, algorithm: str = 'kmeans') -> Dict[str, Any]:
        if not self.ml_available:
            return {'error': 'Machine learning not available. Please install scikit-learn.'}

        try:
            from sklearn.metrics import silhouette_score

            features = self._prepare_features(df, feature_columns)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            n_clusters = min(n_clusters, len(df))
            if n_clusters < 2:
                return {'error': 'Need at least 2 rows to cluster'}

            if algorithm == 'dbscan':
                model = DBSCAN()
                labels = model.fit_predict(scaled)
            elif algorithm == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(scaled)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(scaled)

            unique_labels = set(labels)
            silhouette = (
                float(silhouette_score(scaled, labels))
                if len(unique_labels) > 1 and -1 not in unique_labels
                else None
            )

            cluster_summary = {}
            for label in sorted(unique_labels):
                mask = labels == label
                cluster_summary[f'cluster_{label}'] = {
                    'size': int(mask.sum()),
                    'avg_values': features[mask].mean(numeric_only=True).to_dict(),
                }

            return {
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'feature_columns': feature_columns,
                'cluster_labels': labels.tolist(),
                'silhouette_score': silhouette,
                'cluster_summary': cluster_summary,
            }
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return {'error': str(e)}

    def detect_anomalies(self, df: pd.DataFrame, feature_columns: List[str],
                        method: str = 'isolation_forest') -> Dict[str, Any]:
        if not self.ml_available:
            return {'error': 'Machine learning not available. Please install scikit-learn.'}

        try:
            features = self._prepare_features(df, feature_columns)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            if method == 'elliptic':
                detector = EllipticEnvelope(contamination=0.05, random_state=42)
            else:
                detector = IsolationForest(contamination=0.05, random_state=42)

            predictions = detector.fit_predict(scaled)
            anomaly_mask = predictions == -1
            scores = detector.decision_function(scaled) if hasattr(detector, 'decision_function') else predictions.astype(float)
            indices = df.index[anomaly_mask].tolist()

            return {
                'method': method,
                'feature_columns': feature_columns,
                'anomaly_count': int(anomaly_mask.sum()),
                'anomaly_percentage': round((anomaly_mask.sum() / len(df)) * 100, 2),
                'anomaly_indices': indices,
                'anomaly_scores': [float(scores[i]) for i in range(len(scores)) if anomaly_mask[i]],
            }
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {'error': str(e)}

    def correlation_analysis(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        try:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                return {'error': f'Columns not found: {missing}'}

            numeric = df[columns].apply(pd.to_numeric, errors='coerce')
            corr = numeric.corr()
            matrix = corr.fillna(0).to_dict()

            strong = []
            for i, col1 in enumerate(columns):
                for col2 in columns[i + 1:]:
                    value = corr.loc[col1, col2]
                    if pd.notna(value) and abs(value) >= 0.7:
                        strong.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': round(float(value), 4),
                        })

            recommendations = []
            if strong:
                pair = strong[0]
                recommendations.append(
                    f"Strong correlation ({pair['correlation']}) between '{pair['column1']}' and '{pair['column2']}'"
                )
            else:
                recommendations.append('No very strong correlations (|r| >= 0.7) detected')

            return {
                'columns': columns,
                'correlation_matrix': matrix,
                'strong_correlations': strong,
                'recommendations': recommendations,
            }
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def dimensionality_reduction(self, sheet_name: str, feature_columns: List[str], 
                               n_components: int = 2, method: str = 'pca') -> Dict[str, Any]:
        """
        Perform dimensionality reduction
        """
        if not self.ml_available:
            return {'error': 'Machine learning not available. Please install scikit-learn.'}
        
        try:
            # Placeholder dimensionality reduction results
            return {
                'method': method,
                'original_dimensions': len(feature_columns),
                'reduced_dimensions': n_components,
                'explained_variance_ratio': [0.45, 0.32],
                'cumulative_variance': 0.77,
                'transformed_data': [],
                'recommendations': [
                    f"First {n_components} components explain {77}% of variance",
                    "Consider using fewer features for better interpretability"
                ]
            }
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {str(e)}")
            return {'error': str(e)}
    
    def time_series_analysis(self, sheet_name: str, date_column: str, 
                           value_column: str) -> Dict[str, Any]:
        """
        Perform time series analysis
        """
        try:
            # Placeholder time series results
            return {
                'date_column': date_column,
                'value_column': value_column,
                'trend': 'increasing',
                'seasonality': 'monthly',
                'forecast': {
                    'next_period': 1250.5,
                    'confidence_interval': [1180.2, 1320.8]
                },
                'statistics': {
                    'mean': 1150.3,
                    'std': 85.7,
                    'min': 980.1,
                    'max': 1350.9
                },
                'recommendations': [
                    "Data shows an upward trend",
                    "Consider seasonal adjustments for better forecasting"
                ]
            }
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            return {'error': str(e)}
    
    def get_ml_recommendations(self, sheet_name: str, data_summary: Dict[str, Any]) -> List[str]:
        """
        Generate ML recommendations based on data characteristics
        """
        recommendations = []
        
        if not self.ml_available:
            recommendations.append("Install scikit-learn to enable machine learning features")
            return recommendations
        
        # Analyze data characteristics
        numeric_columns = []
        categorical_columns = []
        
        for sheet_name, sheet_info in data_summary.get('sheets', {}).items():
            for col, dtype in sheet_info.get('data_types', {}).items():
                if 'int' in str(dtype) or 'float' in str(dtype):
                    numeric_columns.append(col)
                else:
                    categorical_columns.append(col)
        
        # Generate recommendations
        if len(numeric_columns) >= 3:
            recommendations.append("Multiple numeric columns detected - consider regression analysis")
        
        if len(categorical_columns) >= 2:
            recommendations.append("Categorical data available - classification models recommended")
        
        if len(numeric_columns) >= 5:
            recommendations.append("High-dimensional data - consider dimensionality reduction")
        
        if data_summary.get('total_rows', 0) > 1000:
            recommendations.append("Large dataset - ensemble methods would work well")
        
        if len(numeric_columns) >= 2:
            recommendations.append("Correlation analysis recommended for numeric features")
        
        return recommendations
    
    def _auto_select_model(self, X_train: np.ndarray, y_train: np.ndarray, is_classification: bool) -> Tuple[Any, float]:
        """
        Automatically select the best model based on data characteristics
        """
        task_type = 'classification' if is_classification else 'regression'
        best_model = None
        best_score = -np.inf
        
        # Try different model categories
        for category, models in self.model_registry[task_type].items():
            for model_class in models:
                try:
                    model = model_class()
                    model.fit(X_train, y_train)
                    score = model.score(X_train, y_train)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception as e:
                    logger.warning(f"Model {model_class.__name__} failed: {e}")
                    continue
        
        return best_model, best_score
    
    def _get_model_by_type(self, model_type: str, is_classification: bool) -> Any:
        """
        Get a specific model by type
        """
        task_type = 'classification' if is_classification else 'regression'
        
        model_map = {
            'random_forest': RandomForestClassifier if is_classification else RandomForestRegressor,
            'linear': LogisticRegression if is_classification else LinearRegression,
            'svm': SVC if is_classification else SVR,
            'neural_network': MLPClassifier if is_classification else MLPRegressor,
            'gradient_boosting': GradientBoostingClassifier if is_classification else GradientBoostingRegressor,
            'decision_tree': DecisionTreeClassifier if is_classification else DecisionTreeRegressor,
            'knn': KNeighborsClassifier if is_classification else KNeighborsRegressor,
            'naive_bayes': GaussianNB if is_classification else None
        }
        
        if model_type in model_map and model_map[model_type]:
            return model_map[model_type]()
        else:
            # Default to random forest
            return RandomForestClassifier() if is_classification else RandomForestRegressor()
    
    def _generate_model_insights(self, df: pd.DataFrame, target_column: str, feature_columns: List[str],
                             metrics: Dict, feature_importance: Dict, is_classification: bool) -> List[str]:
        """Generate rule-based insights about the model and data."""
        insights = []
        
        # Model performance insights
        if is_classification:
            accuracy = metrics.get('accuracy', 0)
            if accuracy > 0.9:
                insights.append("🎯 Excellent model performance! The model achieves over 90% accuracy.")
            elif accuracy > 0.8:
                insights.append("✅ Good model performance with over 80% accuracy.")
            elif accuracy > 0.7:
                insights.append("⚠️ Moderate model performance. Consider feature engineering or more data.")
            else:
                insights.append("❌ Poor model performance. The data may not be suitable for this task.")
        else:
            r2 = metrics.get('r2_score', 0)
            if r2 > 0.8:
                insights.append("🎯 Excellent regression model! R² > 0.8 indicates strong predictive power.")
            elif r2 > 0.6:
                insights.append("✅ Good regression model with decent predictive power.")
            elif r2 > 0.4:
                insights.append("⚠️ Moderate regression model. Consider feature selection or transformation.")
            else:
                insights.append("❌ Poor regression model. The features may not be predictive of the target.")
        
        # Feature importance insights
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append(f"🔍 Top predictive features: {', '.join([f[0] for f in top_features])}")
            
            # Check for feature dominance
            max_importance = max(feature_importance.values())
            if max_importance > 0.5:
                dominant_feature = max(feature_importance.items(), key=lambda x: x[1])[0]
                insights.append(f"⚠️ Feature '{dominant_feature}' dominates the model. Consider feature engineering.")
        
        # Data quality insights
        null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if null_percentage > 10:
            insights.append(f"⚠️ High missing data rate ({null_percentage:.1f}%). Consider imputation strategies.")
        
        # Feature correlation insights
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                insights.append("🔗 High correlation detected between features. Consider removing redundant features.")
        
        # Sample size insights
        if len(df) < 100:
            insights.append("📊 Small dataset detected. Consider collecting more data for better model performance.")
        elif len(df) > 10000:
            insights.append("📊 Large dataset detected. Consider using more sophisticated models or feature selection.")
        
        # Target distribution insights
        if is_classification:
            target_counts = df[target_column].value_counts()
            if len(target_counts) > 10:
                insights.append("⚠️ High number of classes detected. Consider grouping similar classes.")
            elif len(target_counts) == 2:
                class_balance = min(target_counts) / max(target_counts)
                if class_balance < 0.3:
                    insights.append("⚠️ Class imbalance detected. Consider using SMOTE or class weights.")
        
        return insights
    
    def generate_ai_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered data summary
        """
        summary = {
            'data_overview': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'dtypes': df.dtypes.to_dict()
            },
            'quality_metrics': {
                'null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'unique_rows': len(df.drop_duplicates())
            },
            'ai_recommendations': [],
            'potential_issues': [],
            'optimization_suggestions': []
        }
        
        # AI-powered recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 3:
            summary['ai_recommendations'].append("Multiple numeric features detected - ideal for regression analysis")
        
        if len(categorical_cols) >= 2:
            summary['ai_recommendations'].append("Categorical features available - consider classification tasks")
        
        if len(numeric_cols) >= 5:
            summary['ai_recommendations'].append("High-dimensional data - PCA or feature selection recommended")
        
        # Potential issues
        if summary['quality_metrics']['null_percentage'] > 20:
            summary['potential_issues'].append("High missing data rate - data cleaning required")
        
        if summary['quality_metrics']['duplicate_rows'] > len(df) * 0.1:
            summary['potential_issues'].append("Significant duplicate rows detected")
        
        # Optimization suggestions
        if df.memory_usage(deep=True).sum() > 100 * 1024 * 1024:  # 100MB
            summary['optimization_suggestions'].append("Large memory usage - consider data type optimization")
        
        if len(df) > 100000:
            summary['optimization_suggestions'].append("Large dataset - consider sampling for initial analysis")
        
        return summary


if __name__ == "__main__":
    import sys
    from excel_processor import _processor_for_session

    if len(sys.argv) < 2:
        print("Usage: python ml_processor.py <command> [args...]")
        sys.exit(1)

    command = sys.argv[1]
    processor = MLProcessor()

    try:
        if command == "predict":
            session_id, sheet_name = sys.argv[2], sys.argv[3]
            target_column, feature_columns, model_type = sys.argv[4], json.loads(sys.argv[5]), sys.argv[6]
            df = _processor_for_session(session_id).dataframes[sheet_name]
            result = processor.predict_values(df, target_column, feature_columns, model_type)
            print(json.dumps(result, default=str))

        elif command == "cluster":
            session_id, sheet_name = sys.argv[2], sys.argv[3]
            feature_columns = json.loads(sys.argv[4])
            n_clusters = int(sys.argv[5])
            algorithm = sys.argv[6] if len(sys.argv) > 6 else 'kmeans'
            df = _processor_for_session(session_id).dataframes[sheet_name]
            result = processor.cluster_data(df, feature_columns, n_clusters, algorithm)
            print(json.dumps(result, default=str))

        elif command == "anomalies":
            session_id, sheet_name = sys.argv[2], sys.argv[3]
            feature_columns = json.loads(sys.argv[4])
            method = sys.argv[5] if len(sys.argv) > 5 else 'isolation_forest'
            df = _processor_for_session(session_id).dataframes[sheet_name]
            result = processor.detect_anomalies(df, feature_columns, method)
            print(json.dumps(result, default=str))

        elif command == "correlation":
            session_id, sheet_name = sys.argv[2], sys.argv[3]
            columns = json.loads(sys.argv[4])
            df = _processor_for_session(session_id).dataframes[sheet_name]
            result = processor.correlation_analysis(df, columns)
            print(json.dumps(result, default=str))

        elif command == "recommendations":
            session_id, sheet_name = sys.argv[2], sys.argv[3]
            data_summary = json.loads(sys.argv[4])
            result = processor.get_ml_recommendations(sheet_name, data_summary)
            print(json.dumps(result, default=str))

        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
