import pytest
import asyncio
import tempfile
import os
import sys
import pandas as pd
import logging

# Add the api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_excel_data():
    """Create sample Excel data for testing"""
    return pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'Salary': [50000, 60000, 70000, 55000, 65000],
        'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
        'Start_Date': ['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-28', '2020-11-05']
    })

@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    return pd.DataFrame({
        'Product': ['A', 'B', 'C', 'D', 'E'],
        'Price': [10, 20, 30, 40, 50],
        'Quantity': [100, 200, 300, 400, 500],
        'Revenue': [1000, 4000, 9000, 16000, 25000]
    })

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = []
    base_price = 100
    for i, date in enumerate(dates):
        price_change = np.random.normal(0, 2)
        base_price += price_change
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'close_price': max(base_price, 1),
            'open_price': max(base_price - np.random.normal(0, 1), 1),
            'high_price': max(base_price + abs(np.random.normal(0, 2)), 1),
            'low_price': max(base_price - abs(np.random.normal(0, 2)), 1),
            'volume': int(np.random.uniform(1000, 10000))
        })
    
    return data

@pytest.fixture
def temp_excel_file(sample_excel_data):
    """Create temporary Excel file"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        sample_excel_data.to_excel(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create temporary CSV file"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        sample_csv_data.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def temp_json_file(sample_excel_data):
    """Create temporary JSON file"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        sample_excel_data.to_json(tmp.name, orient='records')
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def mock_excel_processor():
    """Create mock Excel processor"""
    from excel_processor import ExcelProcessor
    
    processor = ExcelProcessor()
    
    # Add some test data
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    processor.dataframes['TestSheet'] = test_data
    
    return processor

@pytest.fixture
def test_session_id():
    """Generate test session ID"""
    import uuid
    return str(uuid.uuid4())

@pytest.fixture
def test_file_id():
    """Generate test file ID"""
    import uuid
    return str(uuid.uuid4())

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers",
        "agentci: agentic CI tests that require a reachable local Ollama "
        "(deselected by default via make test-py)",
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker to tests that don't have any marker
        known = {"slow", "integration", "unit", "agentci"}
        if not any(marker.name in known for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

# Test data generators
@pytest.fixture
def large_dataset():
    """Create large dataset for performance testing"""
    import numpy as np
    
    np.random.seed(42)
    n_rows = 10000
    n_cols = 20
    
    data = {}
    for i in range(n_cols):
        data[f'col_{i}'] = np.random.randn(n_rows)
    
    return pd.DataFrame(data)

@pytest.fixture
def dataset_with_nulls():
    """Create dataset with null values"""
    data = pd.DataFrame({
        'A': [1, 2, None, 4, 5, None, 7, 8],
        'B': [None, 2, 3, None, 5, 6, None, 8],
        'C': [1, None, 3, 4, None, 6, 7, None],
        'D': [1, 2, 3, 4, 5, 6, 7, 8]  # No nulls
    })
    return data

@pytest.fixture
def dataset_with_duplicates():
    """Create dataset with duplicates"""
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 3, 4, 5],
        'B': ['x', 'y', 'z', 'x', 'y', 'z', 'a', 'b'],
        'C': [10, 20, 30, 10, 20, 30, 40, 50]
    })
    return data

@pytest.fixture
def dataset_with_outliers():
    """Create dataset with outliers"""
    import numpy as np
    
    np.random.seed(42)
    data = pd.DataFrame({
        'normal': np.random.normal(100, 10, 100),
        'with_outliers': np.concatenate([
            np.random.normal(100, 10, 95),
            np.array([1000, 2000, -1000])  # Outliers
        ])
    })
    return data

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    start_time = time.time()
    yield lambda: time.time() - start_time

@pytest.fixture
def memory_profiler():
    """Memory profiler for testing"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    def get_memory_usage():
        return process.memory_info().rss - initial_memory
    
    return get_memory_usage

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files"""
    temp_files = []
    
    def add_temp_file(file_path):
        temp_files.append(file_path)
    
    yield add_temp_file
    
    # Cleanup
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass  # Ignore cleanup errors

# Test utilities
class TestUtils:
    """Utility functions for tests"""
    
    @staticmethod
    def create_test_excel_file(data, filename=None):
        """Create test Excel file"""
        if filename is None:
            filename = tempfile.mktemp(suffix='.xlsx')
        
        data.to_excel(filename, index=False)
        return filename
    
    @staticmethod
    def create_test_csv_file(data, filename=None):
        """Create test CSV file"""
        if filename is None:
            filename = tempfile.mktemp(suffix='.csv')
        
        data.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def assert_dataframe_equals(df1, df2, **kwargs):
        """Assert two DataFrames are equal"""
        pd.testing.assert_frame_equal(df1, df2, **kwargs)
    
    @staticmethod
    def assert_dataframe_almost_equals(df1, df2, **kwargs):
        """Assert two DataFrames are almost equal"""
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, **kwargs)

@pytest.fixture
def test_utils():
    """Provide test utilities"""
    return TestUtils
