import pandas as pd
import numpy as np
import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# AI/LLM Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. AI assistant features will be disabled.")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic not available. Claude features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Assistant:
    """
    AI-powered assistant for natural language data queries and analysis
    """
    
    def __init__(self):
        self.openai_available = OPENAI_AVAILABLE
        self.anthropic_available = ANTHROPIC_AVAILABLE
        
        # Initialize AI clients
        if self.openai_available:
            self.openai_client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        
        if self.anthropic_available:
            self.anthropic_client = Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
        
        # Query patterns for common data operations
        self.query_patterns = {
            'summary': [
                r'summarize', r'summary', r'overview', r'describe', r'tell me about'
            ],
            'analysis': [
                r'analyze', r'analysis', r'find patterns', r'correlation', r'relationship'
            ],
            'prediction': [
                r'predict', r'forecast', r'what will happen', r'future', r'trend'
            ],
            'filter': [
                r'filter', r'show only', r'where', r'find rows', r'select'
            ],
            'aggregation': [
                r'sum', r'average', r'mean', r'count', r'total', r'max', r'min'
            ],
            'visualization': [
                r'chart', r'graph', r'plot', r'visualize', r'show'
            ],
            'cleaning': [
                r'clean', r'missing', r'duplicate', r'outlier', r'anomaly'
            ]
        }
    
    def process_natural_language_query(self, query: str, df: pd.DataFrame, 
                                     context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process natural language queries about data
        """
        try:
            # Determine query intent
            intent = self._classify_query_intent(query)
            
            # Generate response based on intent
            if intent == 'summary':
                return self._generate_data_summary(query, df, context)
            elif intent == 'analysis':
                return self._generate_analysis(query, df, context)
            elif intent == 'prediction':
                return self._generate_prediction_suggestion(query, df, context)
            elif intent == 'filter':
                return self._generate_filter_suggestion(query, df, context)
            elif intent == 'aggregation':
                return self._generate_aggregation_suggestion(query, df, context)
            elif intent == 'visualization':
                return self._generate_visualization_suggestion(query, df, context)
            elif intent == 'cleaning':
                return self._generate_cleaning_suggestion(query, df, context)
            else:
                return self._generate_general_response(query, df, context)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'response': 'I encountered an error processing your query. Please try rephrasing it.'
            }
    
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the intent of a natural language query
        """
        query_lower = query.lower()
        
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general'
    
    def _generate_data_summary(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered data summary
        """
        summary = {
            'intent': 'summary',
            'response': '',
            'data_insights': {},
            'recommendations': []
        }
        
        # Basic data statistics
        summary['data_insights'] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Generate AI response
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Here's the data summary:
            - Dataset shape: {df.shape}
            - Columns: {df.columns.tolist()}
            - Data types: {df.dtypes.to_dict()}
            - Missing values: {df.isnull().sum().to_dict()}
            
            Provide a helpful, conversational summary of this data. Focus on key insights and what the user can do with this data.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                summary['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                summary['response'] = self._generate_fallback_summary(df)
        else:
            summary['response'] = self._generate_fallback_summary(df)
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(df)
        
        return summary
    
    def _generate_analysis(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered data analysis
        """
        analysis = {
            'intent': 'analysis',
            'response': '',
            'analysis_results': {},
            'insights': []
        }
        
        # Perform basic analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        analysis_results = {}
        
        if len(numeric_cols) > 0:
            analysis_results['numeric_analysis'] = df[numeric_cols].describe().to_dict()
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                analysis_results['correlations'] = df[numeric_cols].corr().to_dict()
        
        if len(categorical_cols) > 0:
            analysis_results['categorical_analysis'] = {}
            for col in categorical_cols:
                analysis_results['categorical_analysis'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].value_counts().head(5).to_dict()
                }
        
        analysis['analysis_results'] = analysis_results
        
        # Generate AI insights
        if self.openai_available:
            prompt = f"""
            You are a web based data analyst AI assistant who only responds in text. A user asked: "{query}"
            
            Here's the analysis results:
            {json.dumps(analysis_results, indent=2)}
            
            Provide insights and patterns you notice in this data. Be specific and actionable.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                analysis['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                analysis['response'] = self._generate_fallback_analysis(analysis_results)
        else:
            analysis['response'] = self._generate_fallback_analysis(analysis_results)
        
        return analysis
    
    def _generate_prediction_suggestion(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered prediction suggestions
        """
        suggestion = {
            'intent': 'prediction',
            'response': '',
            'prediction_suggestions': [],
            'feature_recommendations': []
        }
        
        # Analyze data for prediction potential
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 2:
            suggestion['prediction_suggestions'].append({
                'type': 'regression',
                'description': 'Predict numeric values using regression models',
                'potential_targets': numeric_cols.tolist(),
                'potential_features': numeric_cols.tolist()
            })
        
        if len(categorical_cols) >= 1:
            suggestion['prediction_suggestions'].append({
                'type': 'classification',
                'description': 'Classify categorical outcomes',
                'potential_targets': categorical_cols.tolist(),
                'potential_features': numeric_cols.tolist() if len(numeric_cols) > 0 else []
            })
        
        # Generate AI response
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Dataset info:
            - Numeric columns: {numeric_cols.tolist()}
            - Categorical columns: {categorical_cols.tolist()}
            - Dataset size: {df.shape}
            
            Suggest what predictions are possible with this data and recommend the best approach.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                suggestion['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                suggestion['response'] = self._generate_fallback_prediction_suggestion(df)
        else:
            suggestion['response'] = self._generate_fallback_prediction_suggestion(df)
        
        return suggestion
    
    def _generate_filter_suggestion(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered filter suggestions
        """
        suggestion = {
            'intent': 'filter',
            'response': '',
            'filter_suggestions': [],
            'sample_filters': []
        }
        
        # Generate filter suggestions based on data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            col_stats = df[col].describe()
            suggestion['sample_filters'].append({
                'column': col,
                'type': 'numeric',
                'examples': [
                    f"{col} > {col_stats['25%']:.2f}",
                    f"{col} < {col_stats['75%']:.2f}",
                    f"{col} == {col_stats['mean']:.2f}"
                ]
            })
        
        for col in categorical_cols:
            unique_values = df[col].unique()[:5]  # Top 5 unique values
            suggestion['sample_filters'].append({
                'column': col,
                'type': 'categorical',
                'examples': [f"{col} == '{val}'" for val in unique_values]
            })
        
        # Generate AI response
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Dataset columns: {df.columns.tolist()}
            Dataset shape: {df.shape}
            
            Suggest how to filter this data based on the user's request. Provide specific filter examples.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                suggestion['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                suggestion['response'] = self._generate_fallback_filter_suggestion(df)
        else:
            suggestion['response'] = self._generate_fallback_filter_suggestion(df)
        
        return suggestion
    
    def _generate_aggregation_suggestion(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered aggregation suggestions
        """
        suggestion = {
            'intent': 'aggregation',
            'response': '',
            'aggregation_suggestions': [],
            'sample_aggregations': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Generate aggregation suggestions
        for col in numeric_cols:
            suggestion['sample_aggregations'].append({
                'column': col,
                'operations': ['sum', 'mean', 'median', 'std', 'min', 'max', 'count']
            })
        
        for col in categorical_cols:
            suggestion['sample_aggregations'].append({
                'column': col,
                'operations': ['count', 'nunique', 'mode']
            })
        
        # Generate AI response
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Numeric columns: {numeric_cols.tolist()}
            Categorical columns: {categorical_cols.tolist()}
            
            Suggest appropriate aggregation operations for this data.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                suggestion['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                suggestion['response'] = self._generate_fallback_aggregation_suggestion(df)
        else:
            suggestion['response'] = self._generate_fallback_aggregation_suggestion(df)
        
        return suggestion
    
    def _generate_visualization_suggestion(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered visualization suggestions
        """
        suggestion = {
            'intent': 'visualization',
            'response': '',
            'chart_suggestions': [],
            'recommended_charts': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Generate chart suggestions
        if len(numeric_cols) >= 1:
            suggestion['chart_suggestions'].append({
                'type': 'histogram',
                'description': 'Distribution of numeric data',
                'columns': numeric_cols.tolist()
            })
        
        if len(numeric_cols) >= 2:
            suggestion['chart_suggestions'].append({
                'type': 'scatter',
                'description': 'Relationship between two numeric variables',
                'columns': numeric_cols.tolist()
            })
        
        if len(categorical_cols) >= 1:
            suggestion['chart_suggestions'].append({
                'type': 'bar',
                'description': 'Count of categorical values',
                'columns': categorical_cols.tolist()
            })
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            suggestion['chart_suggestions'].append({
                'type': 'box',
                'description': 'Distribution of numeric data by category',
                'columns': {'numeric': numeric_cols.tolist(), 'categorical': categorical_cols.tolist()}
            })
        
        # Generate AI response
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Numeric columns: {numeric_cols.tolist()}
            Categorical columns: {categorical_cols.tolist()}
            Dataset shape: {df.shape}
            
            Suggest the best visualizations for this data and explain why.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                suggestion['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                suggestion['response'] = self._generate_fallback_visualization_suggestion(df)
        else:
            suggestion['response'] = self._generate_fallback_visualization_suggestion(df)
        
        return suggestion
    
    def _generate_cleaning_suggestion(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered data cleaning suggestions
        """
        suggestion = {
            'intent': 'cleaning',
            'response': '',
            'cleaning_suggestions': [],
            'data_quality_issues': []
        }
        
        # Identify data quality issues
        null_counts = df.isnull().sum()
        duplicate_count = df.duplicated().sum()
        
        if null_counts.sum() > 0:
            suggestion['data_quality_issues'].append({
                'type': 'missing_values',
                'description': 'Missing values detected',
                'columns': null_counts[null_counts > 0].to_dict(),
                'suggestions': ['Fill with mean/median', 'Drop rows/columns', 'Forward/backward fill']
            })
        
        if duplicate_count > 0:
            suggestion['data_quality_issues'].append({
                'type': 'duplicates',
                'description': f'{duplicate_count} duplicate rows found',
                'suggestions': ['Remove duplicates', 'Investigate duplicates']
            })
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                suggestion['data_quality_issues'].append({
                    'type': 'outliers',
                    'description': f'Outliers detected in {col}',
                    'count': len(outliers),
                    'suggestions': ['Remove outliers', 'Cap outliers', 'Transform data']
                })
        
        # Generate AI response
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Data quality issues found:
            {json.dumps(suggestion['data_quality_issues'], indent=2)}
            
            Suggest specific cleaning steps for this data.
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                suggestion['response'] = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                suggestion['response'] = self._generate_fallback_cleaning_suggestion(df)
        else:
            suggestion['response'] = self._generate_fallback_cleaning_suggestion(df)
        
        return suggestion
    
    def _generate_general_response(self, query: str, df: pd.DataFrame, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate general AI response for unclear queries
        """
        response = {
            'intent': 'general',
            'response': '',
            'suggestions': []
        }
        
        if self.openai_available:
            prompt = f"""
            You are a data analyst AI assistant. A user asked: "{query}"
            
            Dataset info:
            - Shape: {df.shape}
            - Columns: {df.columns.tolist()}
            - Data types: {df.dtypes.to_dict()}
            
            Provide helpful suggestions for what they can do with this data.
            """
            
            try:
                ai_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                response['response'] = ai_response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                response['response'] = self._generate_fallback_general_response(df)
        else:
            response['response'] = self._generate_fallback_general_response(df)
        
        return response
    
    # Fallback methods when AI services are not available
    def _generate_fallback_summary(self, df: pd.DataFrame) -> str:
        return f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns. The columns are: {', '.join(df.columns.tolist())}. You can analyze, visualize, or clean this data using the available tools."
    
    def _generate_fallback_analysis(self, analysis_results: Dict) -> str:
        return "I've analyzed your data and found patterns in the numeric and categorical columns. Check the analysis results for detailed statistics and correlations."
    
    def _generate_fallback_prediction_suggestion(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        return f"With {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns, you can perform regression or classification tasks."
    
    def _generate_fallback_filter_suggestion(self, df: pd.DataFrame) -> str:
        return f"You can filter this data by any of the {len(df.columns)} columns. Use conditions like 'column > value' or 'column == value'."
    
    def _generate_fallback_aggregation_suggestion(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return f"You can aggregate numeric columns ({', '.join(numeric_cols.tolist())}) using sum, mean, count, etc."
    
    def _generate_fallback_visualization_suggestion(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        return f"Recommended charts: histograms for numeric data, bar charts for categorical data, and scatter plots for relationships."
    
    def _generate_fallback_cleaning_suggestion(self, df: pd.DataFrame) -> str:
        null_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        return f"Data quality issues: {null_count} missing values and {duplicate_count} duplicate rows. Consider cleaning these issues."
    
    def _generate_fallback_general_response(self, df: pd.DataFrame) -> str:
        return f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns. You can analyze, visualize, clean, or predict with this data."
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate general recommendations based on data characteristics"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 3:
            recommendations.append("Consider correlation analysis for numeric features")
        
        if len(categorical_cols) >= 2:
            recommendations.append("Analyze categorical distributions and relationships")
        
        if df.isnull().sum().sum() > 0:
            recommendations.append("Address missing values before analysis")
        
        if df.duplicated().sum() > 0:
            recommendations.append("Remove duplicate rows for cleaner analysis")
        
        return recommendations


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ai_assistant.py <query> <data_json>")
        sys.exit(1)
    
    query = sys.argv[1]
    data_json = sys.argv[2]
    
    try:
        # Parse data from JSON
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        # Initialize AI assistant
        assistant = Assistant()
        
        # Process query
        result = assistant.process_natural_language_query(query, df)
        
        # Output result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
