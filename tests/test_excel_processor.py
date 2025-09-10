import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from excel_processor import ExcelProcessor, AdvancedStockPredictor

class TestExcelProcessor:
    """Test suite for ExcelProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create ExcelProcessor instance for testing"""
        return ExcelProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample Excel data for testing"""
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Age': [25, 30, 35, 28, 32],
            'Salary': [50000, 60000, 70000, 55000, 65000],
            'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
            'Start_Date': ['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-28', '2020-11-05']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def excel_file(self, sample_data):
        """Create temporary Excel file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            sample_data.to_excel(tmp.name, index=False)
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def csv_file(self, sample_data):
        """Create temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_data.to_csv(tmp.name, index=False)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_load_excel_file(self, processor, excel_file):
        """Test loading Excel file"""
        result = processor.load_file(excel_file)
        
        assert result['total_sheets'] == 1
        assert 'Sheet1' in result['sheets']
        assert result['sheets']['Sheet1']['rows'] == 5
        assert result['sheets']['Sheet1']['columns'] == 5
        assert 'Name' in result['sheets']['Sheet1']['column_names']
    
    def test_load_csv_file(self, processor, csv_file):
        """Test loading CSV file"""
        result = processor.load_file(csv_file)
        
        assert result['total_sheets'] == 1
        assert 'Sheet1' in result['sheets']
        assert result['sheets']['Sheet1']['rows'] == 5
        assert result['sheets']['Sheet1']['columns'] == 5
    
    def test_analyze_data(self, processor, sample_data):
        """Test data analysis functionality"""
        processor.dataframes['TestSheet'] = sample_data
        analysis = processor.analyze_data('TestSheet')
        
        assert analysis['basic_info']['rows'] == 5
        assert analysis['basic_info']['columns'] == 5
        assert 'data_quality' in analysis
        assert 'statistics' in analysis
        assert 'patterns' in analysis
        assert 'recommendations' in analysis
    
    def test_clean_data_remove_duplicates(self, processor, sample_data):
        """Test removing duplicates"""
        # Add duplicate row
        sample_data_with_duplicates = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
        processor.dataframes['TestSheet'] = sample_data_with_duplicates
        
        operations = [{'type': 'remove_duplicates'}]
        result = processor.clean_data('TestSheet', operations)
        
        assert result['final_shape'][0] == 5  # Should remove duplicate
        assert result['changes_summary']['rows_removed'] == 1
    
    def test_clean_data_remove_nulls(self, processor):
        """Test removing null values"""
        # Create data with nulls
        data_with_nulls = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [None, 2, 3, 4, None],
            'C': [1, 2, 3, 4, 5]
        })
        processor.dataframes['TestSheet'] = data_with_nulls
        
        operations = [{'type': 'remove_nulls', 'params': {'strategy': 'drop_rows'}}]
        result = processor.clean_data('TestSheet', operations)
        
        # Should remove rows with any null values
        assert result['final_shape'][0] < 5
    
    def test_clean_data_convert_types(self, processor):
        """Test data type conversion"""
        data = pd.DataFrame({
            'numeric_string': ['1', '2', '3', '4', '5'],
            'text': ['a', 'b', 'c', 'd', 'e']
        })
        processor.dataframes['TestSheet'] = data
        
        operations = [{
            'type': 'convert_types',
            'params': {'conversions': {'numeric_string': 'numeric'}}
        }]
        result = processor.clean_data('TestSheet', operations)
        
        assert result['operations_applied'] == ['convert_types']
        assert processor.dataframes['TestSheet']['numeric_string'].dtype in ['int64', 'float64']
    
    def test_create_chart(self, processor, sample_data):
        """Test chart creation"""
        processor.dataframes['TestSheet'] = sample_data
        
        chart_config = {
            'type': 'bar',
            'title': 'Test Chart',
            'x_column': 'Name',
            'y_columns': ['Salary']
        }
        
        chart_data = processor.create_chart('TestSheet', chart_config)
        
        assert isinstance(chart_data, bytes)
        assert len(chart_data) > 0
    
    def test_export_data_csv(self, processor, sample_data):
        """Test CSV export"""
        processor.dataframes['TestSheet'] = sample_data
        
        export_data = processor.export_data('TestSheet', 'csv')
        
        assert isinstance(export_data, bytes)
        assert len(export_data) > 0
    
    def test_export_data_json(self, processor, sample_data):
        """Test JSON export"""
        processor.dataframes['TestSheet'] = sample_data
        
        export_data = processor.export_data('TestSheet', 'json')
        
        assert isinstance(export_data, bytes)
        # Verify it's valid JSON
        json_data = json.loads(export_data.decode('utf-8'))
        assert isinstance(json_data, list)
        assert len(json_data) == 5
    
    def test_apply_formulas(self, processor, sample_data):
        """Test formula application"""
        processor.dataframes['TestSheet'] = sample_data
        
        formulas = {
            'Total': '=SUM(B:B)',  # Sum of Age column
            'Average': '=AVERAGE(C:C)'  # Average of Salary column
        }
        
        result = processor.apply_formulas('TestSheet', formulas)
        
        assert 'Total' in result
        assert 'Average' in result
        assert result['Total']['success'] == True
        assert result['Average']['success'] == True
    
    def test_get_summary(self, processor, sample_data):
        """Test getting data summary"""
        processor.dataframes['Sheet1'] = sample_data
        processor.dataframes['Sheet2'] = sample_data.copy()
        
        summary = processor.get_summary()
        
        assert summary['total_sheets'] == 2
        assert summary['total_rows'] == 10
        assert summary['total_columns'] == 10
        assert 'Sheet1' in summary['sheets']
        assert 'Sheet2' in summary['sheets']
    
    def test_unsupported_file_format(self, processor):
        """Test handling of unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'This is not a supported format')
            tmp.flush()
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                processor.load_file(tmp.name)
            
            os.unlink(tmp.name)
    
    def test_insufficient_data_for_analysis(self, processor):
        """Test analysis with insufficient data"""
        # Create data with less than 50 rows
        small_data = pd.DataFrame({'A': [1, 2, 3]})
        processor.dataframes['TestSheet'] = small_data
        
        with pytest.raises(ValueError, match="Insufficient data for training"):
            processor.prepare_training_data([{'A': 1}, {'A': 2}, {'A': 3}])

class TestAdvancedStockPredictor:
    """Test suite for AdvancedStockPredictor class"""
    
    @pytest.fixture
    def predictor(self):
        """Create AdvancedStockPredictor instance for testing"""
        return AdvancedStockPredictor()
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = []
        base_price = 100
        for i, date in enumerate(dates):
            price_change = np.random.normal(0, 2)
            base_price += price_change
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'close_price': max(base_price, 1),  # Ensure positive prices
                'open_price': max(base_price - np.random.normal(0, 1), 1),
                'high_price': max(base_price + abs(np.random.normal(0, 2)), 1),
                'low_price': max(base_price - abs(np.random.normal(0, 2)), 1),
                'volume': int(np.random.uniform(1000, 10000))
            })
        
        return data
    
    def test_create_features(self, predictor, sample_stock_data):
        """Test feature creation"""
        df = pd.DataFrame(sample_stock_data)
        df_with_features = predictor.create_features(df)
        
        # Check that new features were created
        expected_features = [
            'price_change', 'price_range', 'volume_price_ratio',
            'ma_5', 'ma_10', 'ma_20', 'rsi', 'macd'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
    
    def test_calculate_rsi(self, predictor):
        """Test RSI calculation"""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = predictor._calculate_rsi(prices)
        
        assert len(rsi) == len(prices)
        assert not rsi.isna().all()  # Should have some valid RSI values
    
    def test_train_model(self, predictor, sample_stock_data):
        """Test model training"""
        result = predictor.train_model(sample_stock_data)
        
        assert 'best_model' in result
        assert 'scores' in result
        assert 'training_samples' in result
        assert 'test_samples' in result
        assert predictor.is_trained == True
    
    def test_predict_stock_price(self, predictor, sample_stock_data):
        """Test stock price prediction"""
        # First train the model
        predictor.train_model(sample_stock_data)
        
        # Get the last data point for prediction
        current_data = sample_stock_data[-1]
        
        prediction = predictor.predict_stock_price(current_data)
        
        assert 'predicted_price' in prediction
        assert 'confidence' in prediction
        assert 'prediction_date' in prediction
        assert 'model_type' in prediction
        assert isinstance(prediction['predicted_price'], (int, float))
        assert 0 <= prediction['confidence'] <= 1
    
    def test_save_and_load_model(self, predictor, sample_stock_data, tmp_path):
        """Test model saving and loading"""
        # Train the model
        predictor.train_model(sample_stock_data)
        
        # Save the model
        model_path = tmp_path / "test_model.pkl"
        predictor.save_model(str(model_path))
        
        # Create new predictor and load model
        new_predictor = AdvancedStockPredictor()
        new_predictor.load_model(str(model_path))
        
        assert new_predictor.is_trained == True
        assert len(new_predictor.feature_columns) > 0
    
    def test_model_with_insufficient_data(self, predictor):
        """Test model training with insufficient data"""
        small_data = [{'close_price': 100, 'volume': 1000}] * 10  # Only 10 data points
        
        with pytest.raises(ValueError, match="Insufficient data for training"):
            predictor.train_model(small_data)
    
    def test_prediction_without_training(self, predictor):
        """Test prediction without training"""
        current_data = {'close_price': 100, 'volume': 1000}
        
        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict_stock_price(current_data)

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_excel_processing(self):
        """Test complete Excel processing workflow"""
        processor = ExcelProcessor()
        
        # Create sample data
        data = pd.DataFrame({
            'Product': ['A', 'B', 'C', 'D', 'E'],
            'Price': [10, 20, 30, 40, 50],
            'Quantity': [100, 200, 300, 400, 500],
            'Revenue': [1000, 4000, 9000, 16000, 25000]
        })
        
        # Save to temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            data.to_excel(tmp.name, index=False)
            
            # Load file
            file_info = processor.load_file(tmp.name)
            assert file_info['total_sheets'] == 1
            
            # Analyze data
            analysis = processor.analyze_data('Sheet1')
            assert analysis['basic_info']['rows'] == 5
            
            # Clean data (remove duplicates)
            operations = [{'type': 'remove_duplicates'}]
            clean_result = processor.clean_data('Sheet1', operations)
            assert clean_result['operations_applied'] == ['remove_duplicates']
            
            # Create chart
            chart_config = {
                'type': 'bar',
                'title': 'Revenue Chart',
                'x_column': 'Product',
                'y_columns': ['Revenue']
            }
            chart_data = processor.create_chart('Sheet1', chart_config)
            assert isinstance(chart_data, bytes)
            
            # Export data
            export_data = processor.export_data('Sheet1', 'csv')
            assert isinstance(export_data, bytes)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_batch_processing(self):
        """Test batch processing multiple files"""
        processor = ExcelProcessor()
        
        # Create multiple test files
        files = []
        for i in range(3):
            data = pd.DataFrame({
                'ID': range(i*10, (i+1)*10),
                'Value': range(i*10, (i+1)*10)
            })
            
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                data.to_excel(tmp.name, index=False)
                files.append(tmp.name)
        
        try:
            # Process each file
            results = []
            for file_path in files:
                file_info = processor.load_file(file_path)
                analysis = processor.analyze_data('Sheet1')
                results.append({
                    'file': file_path,
                    'rows': analysis['basic_info']['rows'],
                    'columns': analysis['basic_info']['columns']
                })
            
            # Verify all files were processed
            assert len(results) == 3
            for result in results:
                assert result['rows'] == 10
                assert result['columns'] == 2
                
        finally:
            # Clean up
            for file_path in files:
                os.unlink(file_path)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
