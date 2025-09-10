import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from api_routes import router
from fastapi.testclient import TestClient
from fastapi import FastAPI

class TestAPIRoutes:
    """Test suite for API routes"""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing"""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create sample Excel file for testing"""
        import pandas as pd
        
        data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Salary': [50000, 60000, 70000]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            data.to_excel(tmp.name, index=False)
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create sample CSV file for testing"""
        import pandas as pd
        
        data = pd.DataFrame({
            'Product': ['A', 'B', 'C'],
            'Price': [10, 20, 30],
            'Quantity': [100, 200, 300]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            data.to_csv(tmp.name, index=False)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_upload_excel_file(self, client, sample_excel_file):
        """Test Excel file upload"""
        with open(sample_excel_file, 'rb') as f:
            files = {'file': ('test.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            
            with patch('api_routes.processExcelFile') as mock_process:
                mock_process.return_value = {
                    'file_path': sample_excel_file,
                    'total_sheets': 1,
                    'sheets': {'Sheet1': {'rows': 3, 'columns': 3}},
                    'loaded_at': '2023-01-01T00:00:00'
                }
                
                response = client.post('/excel/upload', files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert data['success'] == True
                assert 'file_info' in data
                assert data['file_info']['total_sheets'] == 1
    
    def test_upload_csv_file(self, client, sample_csv_file):
        """Test CSV file upload"""
        with open(sample_csv_file, 'rb') as f:
            files = {'file': ('test.csv', f, 'text/csv')}
            
            with patch('api_routes.processExcelFile') as mock_process:
                mock_process.return_value = {
                    'file_path': sample_csv_file,
                    'total_sheets': 1,
                    'sheets': {'Sheet1': {'rows': 3, 'columns': 3}},
                    'loaded_at': '2023-01-01T00:00:00'
                }
                
                response = client.post('/excel/upload', files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert data['success'] == True
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'This is not a valid file')
            tmp.flush()
            
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = client.post('/excel/upload', files=files)
                
                assert response.status_code == 400
                data = response.json()
                assert 'error' in data
            
            os.unlink(tmp.name)
    
    def test_upload_no_file(self, client):
        """Test upload without file"""
        response = client.post('/excel/upload')
        
        assert response.status_code == 400
        data = response.json()
        assert data['error']['message'] == 'No file uploaded'
    
    def test_analyze_data(self, client):
        """Test data analysis endpoint"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        
        with patch('api_routes.analyzeSheetData') as mock_analyze:
            mock_analyze.return_value = {
                'basic_info': {'rows': 100, 'columns': 5},
                'data_quality': {'duplicate_rows': 2, 'empty_rows': 1},
                'statistics': {'numeric': {}},
                'patterns': {'outliers': {}},
                'recommendations': ['Remove duplicates']
            }
            
            response = client.get(f'/excel/{session_id}/analyze/{sheet_name}')
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert 'analysis' in data
            assert data['analysis']['basic_info']['rows'] == 100
    
    def test_clean_data(self, client):
        """Test data cleaning endpoint"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        operations = [
            {'type': 'remove_duplicates'},
            {'type': 'remove_nulls', 'params': {'strategy': 'drop_rows'}}
        ]
        
        with patch('api_routes.cleanSheetData') as mock_clean:
            mock_clean.return_value = {
                'original_shape': (100, 5),
                'final_shape': (95, 5),
                'operations_applied': ['remove_duplicates', 'remove_nulls'],
                'changes_summary': {'rows_removed': 5, 'columns_removed': 0}
            }
            
            response = client.post(
                f'/excel/{session_id}/clean/{sheet_name}',
                json={'operations': operations}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert 'result' in data
            assert data['result']['operations_applied'] == ['remove_duplicates', 'remove_nulls']
    
    def test_create_chart(self, client):
        """Test chart creation endpoint"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        chart_config = {
            'type': 'bar',
            'title': 'Test Chart',
            'x_column': 'Name',
            'y_columns': ['Salary']
        }
        
        with patch('api_routes.createChart') as mock_chart:
            mock_chart.return_value = b'fake chart data'
            
            response = client.post(
                f'/excel/{session_id}/chart/{sheet_name}',
                json={'chart_config': chart_config}
            )
            
            assert response.status_code == 200
            assert response.headers['content-type'] == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            assert response.headers['content-disposition'] == f'attachment; filename="chart_{sheet_name}.xlsx"'
    
    def test_export_data_csv(self, client):
        """Test CSV export endpoint"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        
        with patch('api_routes.exportSheetData') as mock_export:
            mock_export.return_value = b'Name,Age,Salary\nAlice,25,50000\nBob,30,60000'
            
            response = client.get(f'/excel/{session_id}/export/{sheet_name}?format=csv')
            
            assert response.status_code == 200
            assert response.headers['content-type'] == 'text/csv'
            assert response.headers['content-disposition'] == f'attachment; filename="{sheet_name}.csv"'
    
    def test_export_data_json(self, client):
        """Test JSON export endpoint"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        
        with patch('api_routes.exportSheetData') as mock_export:
            mock_export.return_value = b'[{"Name":"Alice","Age":25,"Salary":50000}]'
            
            response = client.get(f'/excel/{session_id}/export/{sheet_name}?format=json')
            
            assert response.status_code == 200
            assert response.headers['content-type'] == 'application/json'
            assert response.headers['content-disposition'] == f'attachment; filename="{sheet_name}.json"'
    
    def test_export_data_invalid_format(self, client):
        """Test export with invalid format"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        
        response = client.get(f'/excel/{session_id}/export/{sheet_name}?format=invalid')
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
    
    def test_apply_formulas(self, client):
        """Test formula application endpoint"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        formulas = {
            'Total': '=SUM(A:A)',
            'Average': '=AVERAGE(B:B)'
        }
        
        with patch('api_routes.applyFormulas') as mock_formulas:
            mock_formulas.return_value = {
                'Total': {'success': True, 'formula': '=SUM(A:A)', 'result_type': 'float64'},
                'Average': {'success': True, 'formula': '=AVERAGE(B:B)', 'result_type': 'float64'}
            }
            
            response = client.post(
                f'/excel/{session_id}/formulas/{sheet_name}',
                json={'formulas': formulas}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert 'result' in data
            assert data['result']['Total']['success'] == True
            assert data['result']['Average']['success'] == True
    
    def test_get_summary(self, client):
        """Test data summary endpoint"""
        session_id = 'test-session'
        
        with patch('api_routes.getDataSummary') as mock_summary:
            mock_summary.return_value = {
                'total_sheets': 2,
                'total_rows': 200,
                'total_columns': 10,
                'total_memory_usage': 1024000,
                'sheets': {
                    'Sheet1': {'rows': 100, 'columns': 5},
                    'Sheet2': {'rows': 100, 'columns': 5}
                }
            }
            
            response = client.get(f'/excel/{session_id}/summary')
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert 'summary' in data
            assert data['summary']['total_sheets'] == 2
            assert data['summary']['total_rows'] == 200
    
    def test_batch_processing(self, client, sample_excel_file, sample_csv_file):
        """Test batch processing endpoint"""
        files = []
        
        # Create multiple test files
        for i, file_path in enumerate([sample_excel_file, sample_csv_file]):
            with open(file_path, 'rb') as f:
                files.append(('files', (f'test_{i}.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))
        
        with patch('api_routes.processExcelFile') as mock_process:
            mock_process.return_value = {
                'file_path': 'test.xlsx',
                'total_sheets': 1,
                'sheets': {'Sheet1': {'rows': 3, 'columns': 3}},
                'loaded_at': '2023-01-01T00:00:00'
            }
            
            response = client.post('/excel/batch', files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert 'results' in data
            assert len(data['results']) == 2
    
    def test_batch_processing_no_files(self, client):
        """Test batch processing without files"""
        response = client.post('/excel/batch')
        
        assert response.status_code == 400
        data = response.json()
        assert data['error']['message'] == 'No files uploaded'
    
    def test_validation_errors(self, client):
        """Test validation error handling"""
        # Test with invalid session ID
        response = client.get('/excel//analyze/Sheet1')
        assert response.status_code == 400
        
        # Test with invalid sheet name
        response = client.get('/excel/test-session//analyze/')
        assert response.status_code == 400
        
        # Test with invalid operations
        response = client.post('/excel/test-session/clean/Sheet1', json={'operations': 'invalid'})
        assert response.status_code == 400
    
    def test_error_handling(self, client):
        """Test error handling in API routes"""
        session_id = 'test-session'
        sheet_name = 'Sheet1'
        
        with patch('api_routes.analyzeSheetData') as mock_analyze:
            mock_analyze.side_effect = Exception('Processing error')
            
            response = client.get(f'/excel/{session_id}/analyze/{sheet_name}')
            
            assert response.status_code == 500
            data = response.json()
            assert 'error' in data
    
    def test_file_cleanup_on_error(self, client):
        """Test file cleanup when processing fails"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp.write(b'fake excel data')
            tmp.flush()
            
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                
                with patch('api_routes.processExcelFile') as mock_process:
                    mock_process.side_effect = Exception('Processing failed')
                    
                    response = client.post('/excel/upload', files=files)
                    
                    assert response.status_code == 500
            
            # File should be cleaned up
            assert not os.path.exists(tmp.name)
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('api_routes.getDataSummary') as mock_summary:
                mock_summary.return_value = {'total_sheets': 1}
                
                response = client.get('/excel/test-session/summary')
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

class TestAPIIntegration:
    """Integration tests for API"""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing"""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    def test_complete_workflow(self, client):
        """Test complete Excel processing workflow"""
        import pandas as pd
        import tempfile
        
        # Create test data
        data = pd.DataFrame({
            'Product': ['A', 'B', 'C', 'D', 'E'],
            'Price': [10, 20, 30, 40, 50],
            'Quantity': [100, 200, 300, 400, 500],
            'Revenue': [1000, 4000, 9000, 16000, 25000]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            data.to_excel(tmp.name, index=False)
            
            # Step 1: Upload file
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                
                with patch('api_routes.processExcelFile') as mock_process:
                    mock_process.return_value = {
                        'file_path': tmp.name,
                        'total_sheets': 1,
                        'sheets': {'Sheet1': {'rows': 5, 'columns': 4}},
                        'loaded_at': '2023-01-01T00:00:00'
                    }
                    
                    upload_response = client.post('/excel/upload', files=files)
                    assert upload_response.status_code == 200
            
            # Step 2: Analyze data
            session_id = 'test-session'
            with patch('api_routes.analyzeSheetData') as mock_analyze:
                mock_analyze.return_value = {
                    'basic_info': {'rows': 5, 'columns': 4},
                    'data_quality': {'duplicate_rows': 0, 'empty_rows': 0},
                    'statistics': {'numeric': {}},
                    'patterns': {'outliers': {}},
                    'recommendations': []
                }
                
                analyze_response = client.get(f'/excel/{session_id}/analyze/Sheet1')
                assert analyze_response.status_code == 200
            
            # Step 3: Clean data
            with patch('api_routes.cleanSheetData') as mock_clean:
                mock_clean.return_value = {
                    'original_shape': (5, 4),
                    'final_shape': (5, 4),
                    'operations_applied': ['remove_duplicates'],
                    'changes_summary': {'rows_removed': 0, 'columns_removed': 0}
                }
                
                clean_response = client.post(
                    f'/excel/{session_id}/clean/Sheet1',
                    json={'operations': [{'type': 'remove_duplicates'}]}
                )
                assert clean_response.status_code == 200
            
            # Step 4: Create chart
            with patch('api_routes.createChart') as mock_chart:
                mock_chart.return_value = b'fake chart data'
                
                chart_response = client.post(
                    f'/excel/{session_id}/chart/Sheet1',
                    json={
                        'chart_config': {
                            'type': 'bar',
                            'title': 'Revenue Chart',
                            'x_column': 'Product',
                            'y_columns': ['Revenue']
                        }
                    }
                )
                assert chart_response.status_code == 200
            
            # Step 5: Export data
            with patch('api_routes.exportSheetData') as mock_export:
                mock_export.return_value = b'Product,Price,Quantity,Revenue\nA,10,100,1000'
                
                export_response = client.get(f'/excel/{session_id}/export/Sheet1?format=csv')
                assert export_response.status_code == 200
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_error_recovery(self, client):
        """Test error recovery in workflow"""
        session_id = 'test-session'
        
        # Test with non-existent session
        with patch('api_routes.analyzeSheetData') as mock_analyze:
            mock_analyze.side_effect = Exception('Session not found')
            
            response = client.get(f'/excel/{session_id}/analyze/Sheet1')
            assert response.status_code == 500
        
        # Test with invalid operations
        with patch('api_routes.cleanSheetData') as mock_clean:
            mock_clean.side_effect = Exception('Invalid operation')
            
            response = client.post(
                f'/excel/{session_id}/clean/Sheet1',
                json={'operations': [{'type': 'invalid_operation'}]}
            )
            assert response.status_code == 500

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
