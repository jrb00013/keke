import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from cloud_services import (
    CloudConfig, AWSCloudService, AzureCloudService, GCPCloudService,
    RedisCacheService, DatabaseService, CloudServiceManager, cloud_manager
)

class TestCloudConfig:
    """Test suite for CloudConfig"""
    
    def test_cloud_config_defaults(self):
        """Test CloudConfig with default values"""
        config = CloudConfig()
        
        assert config.aws_access_key is None
        assert config.aws_secret_key is None
        assert config.aws_region == 'us-east-1'
        assert config.azure_connection_string is None
        assert config.gcp_credentials_path is None
        assert config.redis_url == 'redis://localhost:6379'
        assert config.mongodb_url == 'mongodb://localhost:27017'
        assert config.postgres_url == 'postgresql://localhost:5432/keke'
    
    def test_cloud_config_custom_values(self):
        """Test CloudConfig with custom values"""
        config = CloudConfig(
            aws_access_key='test_key',
            aws_secret_key='test_secret',
            aws_region='us-west-2',
            redis_url='redis://test:6379'
        )
        
        assert config.aws_access_key == 'test_key'
        assert config.aws_secret_key == 'test_secret'
        assert config.aws_region == 'us-west-2'
        assert config.redis_url == 'redis://test:6379'

class TestAWSCloudService:
    """Test suite for AWS cloud service"""
    
    @pytest.fixture
    def aws_service(self):
        """Create AWS service instance for testing"""
        config = CloudConfig(
            aws_access_key='test_key',
            aws_secret_key='test_secret',
            aws_region='us-east-1'
        )
        return AWSCloudService(config)
    
    @patch('cloud_services.boto3.Session')
    def test_aws_service_initialization(self, mock_session, aws_service):
        """Test AWS service initialization"""
        mock_session.assert_called_once()
        assert aws_service.config.aws_access_key == 'test_key'
        assert aws_service.config.aws_secret_key == 'test_secret'
    
    @patch('cloud_services.boto3.Session')
    async def test_upload_to_s3(self, mock_session, aws_service):
        """Test S3 upload functionality"""
        mock_s3 = Mock()
        mock_session.return_value.client.return_value = mock_s3
        
        data = b'test data'
        result = await aws_service.upload_to_s3('test-bucket', 'test-key', data)
        
        mock_s3.put_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-key',
            Body=data,
            ContentType='application/octet-stream'
        )
        assert result == True
    
    @patch('cloud_services.boto3.Session')
    async def test_download_from_s3(self, mock_session, aws_service):
        """Test S3 download functionality"""
        mock_s3 = Mock()
        mock_response = Mock()
        mock_response['Body'].read.return_value = b'test data'
        mock_s3.get_object.return_value = mock_response
        mock_session.return_value.client.return_value = mock_s3
        
        result = await aws_service.download_from_s3('test-bucket', 'test-key')
        
        mock_s3.get_object.assert_called_once_with(Bucket='test-bucket', Key='test-key')
        assert result == b'test data'
    
    @patch('cloud_services.boto3.Session')
    async def test_invoke_lambda(self, mock_session, aws_service):
        """Test Lambda invocation"""
        mock_lambda = Mock()
        mock_response = Mock()
        mock_response['Payload'].read.return_value = json.dumps({'result': 'success'})
        mock_lambda.invoke.return_value = mock_response
        mock_session.return_value.client.return_value = mock_lambda
        
        payload = {'test': 'data'}
        result = await aws_service.invoke_lambda('test-function', payload)
        
        mock_lambda.invoke.assert_called_once_with(
            FunctionName='test-function',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        assert result == {'result': 'success'}
    
    @patch('cloud_services.boto3.Session')
    async def test_store_in_dynamodb(self, mock_session, aws_service):
        """Test DynamoDB storage"""
        mock_dynamodb = Mock()
        mock_table = Mock()
        mock_dynamodb.Table.return_value = mock_table
        mock_session.return_value.resource.return_value = mock_dynamodb
        
        item = {'id': 'test', 'data': 'value'}
        result = await aws_service.store_in_dynamodb('test-table', item)
        
        mock_table.put_item.assert_called_once_with(Item=item)
        assert result == True
    
    @patch('cloud_services.boto3.Session')
    async def test_send_sqs_message(self, mock_session, aws_service):
        """Test SQS message sending"""
        mock_sqs = Mock()
        mock_session.return_value.client.return_value = mock_sqs
        
        result = await aws_service.send_sqs_message('test-queue-url', 'test message')
        
        mock_sqs.send_message.assert_called_once_with(
            QueueUrl='test-queue-url',
            MessageBody='test message',
            DelaySeconds=0
        )
        assert result == True
    
    @patch('cloud_services.boto3.Session')
    async def test_publish_sns_message(self, mock_session, aws_service):
        """Test SNS message publishing"""
        mock_sns = Mock()
        mock_session.return_value.client.return_value = mock_sns
        
        result = await aws_service.publish_sns_message('test-topic-arn', 'test message', 'test subject')
        
        mock_sns.publish.assert_called_once_with(
            TopicArn='test-topic-arn',
            Message='test message',
            Subject='test subject'
        )
        assert result == True

class TestRedisCacheService:
    """Test suite for Redis cache service"""
    
    @pytest.fixture
    def redis_service(self):
        """Create Redis service instance for testing"""
        config = CloudConfig(redis_url='redis://localhost:6379')
        return RedisCacheService(config)
    
    @patch('cloud_services.redis.from_url')
    async def test_set_cache(self, mock_redis_from_url, redis_service):
        """Test cache setting"""
        mock_redis = Mock()
        mock_redis_from_url.return_value = mock_redis
        
        result = await redis_service.set_cache('test-key', {'data': 'value'}, 3600)
        
        mock_redis.setex.assert_called_once()
        assert result == True
    
    @patch('cloud_services.redis.from_url')
    async def test_get_cache(self, mock_redis_from_url, redis_service):
        """Test cache retrieval"""
        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({'data': 'value'})
        mock_redis_from_url.return_value = mock_redis
        
        result = await redis_service.get_cache('test-key')
        
        mock_redis.get.assert_called_once_with('test-key')
        assert result == {'data': 'value'}
    
    @patch('cloud_services.redis.from_url')
    async def test_get_cache_miss(self, mock_redis_from_url, redis_service):
        """Test cache miss"""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis_from_url.return_value = mock_redis
        
        result = await redis_service.get_cache('test-key')
        
        assert result is None
    
    @patch('cloud_services.redis.from_url')
    async def test_delete_cache(self, mock_redis_from_url, redis_service):
        """Test cache deletion"""
        mock_redis = Mock()
        mock_redis_from_url.return_value = mock_redis
        
        result = await redis_service.delete_cache('test-key')
        
        mock_redis.delete.assert_called_once_with('test-key')
        assert result == True
    
    @patch('cloud_services.redis.from_url')
    async def test_increment_counter(self, mock_redis_from_url, redis_service):
        """Test counter increment"""
        mock_redis = Mock()
        mock_redis.incrby.return_value = 5
        mock_redis_from_url.return_value = mock_redis
        
        result = await redis_service.increment_counter('test-counter', 2)
        
        mock_redis.incrby.assert_called_once_with('test-counter', 2)
        assert result == 5

class TestDatabaseService:
    """Test suite for database service"""
    
    @pytest.fixture
    def db_service(self):
        """Create database service instance for testing"""
        config = CloudConfig(
            mongodb_url='mongodb://localhost:27017',
            postgres_url='postgresql://localhost:5432/keke'
        )
        return DatabaseService(config)
    
    @patch('cloud_services.pymongo.MongoClient')
    async def test_store_excel_metadata(self, mock_mongo_client, db_service):
        """Test Excel metadata storage in MongoDB"""
        mock_collection = Mock()
        mock_collection.insert_one.return_value.inserted_id = 'test-id'
        mock_db = Mock()
        mock_db.excel_files = mock_collection
        mock_client = Mock()
        mock_client.keke = mock_db
        mock_mongo_client.return_value = mock_client
        
        metadata = {'file_name': 'test.xlsx', 'size': 1024}
        result = await db_service.store_excel_metadata(metadata)
        
        mock_collection.insert_one.assert_called_once_with(metadata)
        assert result == True
    
    @patch('cloud_services.pymongo.MongoClient')
    async def test_get_excel_metadata(self, mock_mongo_client, db_service):
        """Test Excel metadata retrieval from MongoDB"""
        mock_collection = Mock()
        mock_collection.find_one.return_value = {'_id': 'test-id', 'file_name': 'test.xlsx'}
        mock_db = Mock()
        mock_db.excel_files = mock_collection
        mock_client = Mock()
        mock_client.keke = mock_db
        mock_mongo_client.return_value = mock_client
        
        result = await db_service.get_excel_metadata('test-id')
        
        mock_collection.find_one.assert_called_once_with({'_id': 'test-id'})
        assert result == {'_id': 'test-id', 'file_name': 'test.xlsx'}
    
    @patch('cloud_services.create_engine')
    async def test_store_processing_results(self, mock_create_engine, db_service):
        """Test processing results storage in PostgreSQL"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        results = {
            'file_id': 'test-file',
            'operation_type': 'analysis',
            'status': 'completed',
            'data': {'result': 'success'}
        }
        
        result = await db_service.store_processing_results(results)
        
        mock_connection.execute.assert_called_once()
        mock_connection.commit.assert_called_once()
        assert result == True

class TestCloudServiceManager:
    """Test suite for cloud service manager"""
    
    @pytest.fixture
    def cloud_manager(self):
        """Create cloud service manager for testing"""
        config = CloudConfig(
            aws_access_key='test_key',
            aws_secret_key='test_secret',
            azure_connection_string='test_connection',
            gcp_credentials_path='test_credentials.json'
        )
        return CloudServiceManager(config)
    
    def test_cloud_manager_initialization(self, cloud_manager):
        """Test cloud manager initialization"""
        assert cloud_manager.aws is not None
        assert cloud_manager.azure is not None
        assert cloud_manager.gcp is not None
        assert cloud_manager.redis is not None
        assert cloud_manager.database is not None
        
        # Check service availability
        assert cloud_manager.services_available['aws'] == True
        assert cloud_manager.services_available['azure'] == True
        assert cloud_manager.services_available['gcp'] == True
        assert cloud_manager.services_available['redis'] == True
        assert cloud_manager.services_available['database'] == True
    
    @patch.object(AWSCloudService, 'upload_to_s3')
    @patch.object(AzureCloudService, 'upload_to_blob')
    @patch.object(GCPCloudService, 'upload_to_gcs')
    async def test_backup_excel_file(self, mock_gcs_upload, mock_azure_upload, 
                                   mock_s3_upload, cloud_manager):
        """Test Excel file backup to multiple providers"""
        mock_s3_upload.return_value = True
        mock_azure_upload.return_value = True
        mock_gcs_upload.return_value = True
        
        file_data = b'test excel data'
        file_name = 'test.xlsx'
        
        result = await cloud_manager.backup_excel_file(file_data, file_name)
        
        assert result['aws'] == True
        assert result['azure'] == True
        assert result['gcp'] == True
        
        # Verify upload calls
        mock_s3_upload.assert_called_once()
        mock_azure_upload.assert_called_once()
        mock_gcs_upload.assert_called_once()
    
    @patch.object(RedisCacheService, 'set_cache')
    async def test_cache_analysis_results(self, mock_set_cache, cloud_manager):
        """Test analysis results caching"""
        mock_set_cache.return_value = True
        
        file_id = 'test-file'
        analysis = {'result': 'success'}
        
        result = await cloud_manager.cache_analysis_results(file_id, analysis)
        
        mock_set_cache.assert_called_once()
        assert result == True
    
    @patch.object(RedisCacheService, 'get_cache')
    async def test_get_cached_analysis(self, mock_get_cache, cloud_manager):
        """Test cached analysis retrieval"""
        mock_get_cache.return_value = {'result': 'success'}
        
        file_id = 'test-file'
        result = await cloud_manager.get_cached_analysis(file_id)
        
        mock_get_cache.assert_called_once()
        assert result == {'result': 'success'}
    
    @patch.object(DatabaseService, 'store_processing_results')
    @patch.object(AWSCloudService, 'publish_sns_message')
    async def test_log_processing_event(self, mock_sns_publish, mock_db_store, cloud_manager):
        """Test processing event logging"""
        mock_db_store.return_value = True
        mock_sns_publish.return_value = True
        
        event_type = 'file_processed'
        data = {'file_id': 'test-file', 'status': 'success'}
        
        result = await cloud_manager.log_processing_event(event_type, data)
        
        mock_db_store.assert_called_once()
        mock_sns_publish.assert_called_once()
        assert result == True
    
    @patch.object(AWSCloudService, 'put_cloudwatch_metric')
    @patch.object(RedisCacheService, 'set_cache')
    @patch.object(DatabaseService, 'get_processing_history')
    async def test_get_service_status(self, mock_db_query, mock_redis_set, 
                                    mock_cloudwatch_put, cloud_manager):
        """Test service status checking"""
        mock_cloudwatch_put.return_value = True
        mock_redis_set.return_value = True
        mock_db_query.return_value = []
        
        result = await cloud_manager.get_service_status()
        
        assert 'aws' in result
        assert 'redis' in result
        assert 'database' in result
        
        # Check that status checks were performed
        mock_cloudwatch_put.assert_called_once()
        mock_redis_set.assert_called_once()
        mock_db_query.assert_called_once()

class TestIntegration:
    """Integration tests for cloud services"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_cloud_workflow(self):
        """Test complete cloud workflow"""
        config = CloudConfig(
            aws_access_key='test_key',
            aws_secret_key='test_secret'
        )
        
        with patch('cloud_services.boto3.Session') as mock_session:
            # Mock AWS services
            mock_s3 = Mock()
            mock_s3.put_object.return_value = None
            mock_session.return_value.client.return_value = mock_s3
            
            # Mock Redis
            with patch('cloud_services.redis.from_url') as mock_redis:
                mock_redis_client = Mock()
                mock_redis_client.setex.return_value = None
                mock_redis_client.get.return_value = None
                mock_redis.return_value = mock_redis_client
                
                # Mock MongoDB
                with patch('cloud_services.pymongo.MongoClient') as mock_mongo:
                    mock_collection = Mock()
                    mock_collection.insert_one.return_value.inserted_id = 'test-id'
                    mock_db = Mock()
                    mock_db.excel_files = mock_collection
                    mock_client = Mock()
                    mock_client.keke = mock_db
                    mock_mongo.return_value = mock_client
                    
                    # Mock PostgreSQL
                    with patch('cloud_services.create_engine') as mock_engine:
                        mock_engine_instance = Mock()
                        mock_connection = Mock()
                        mock_engine_instance.connect.return_value.__enter__.return_value = mock_connection
                        mock_engine.return_value = mock_engine_instance
                        
                        # Create cloud manager
                        cloud_manager = CloudServiceManager(config)
                        
                        # Test workflow
                        file_data = b'test excel data'
                        file_name = 'test.xlsx'
                        
                        # Backup file
                        backup_result = await cloud_manager.backup_excel_file(file_data, file_name)
                        assert backup_result['aws'] == True
                        
                        # Cache analysis
                        analysis = {'rows': 100, 'columns': 5}
                        cache_result = await cloud_manager.cache_analysis_results('test-file', analysis)
                        assert cache_result == True
                        
                        # Log event
                        event_result = await cloud_manager.log_processing_event('file_processed', {
                            'file_id': 'test-file',
                            'status': 'success'
                        })
                        assert event_result == True
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in cloud services"""
        config = CloudConfig()
        
        with patch('cloud_services.boto3.Session') as mock_session:
            # Mock AWS service with error
            mock_s3 = Mock()
            mock_s3.put_object.side_effect = Exception('S3 error')
            mock_session.return_value.client.return_value = mock_s3
            
            aws_service = AWSCloudService(config)
            
            # Test error handling
            result = await aws_service.upload_to_s3('bucket', 'key', b'data')
            assert result == False
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent cloud operations"""
        config = CloudConfig()
        
        with patch('cloud_services.redis.from_url') as mock_redis:
            mock_redis_client = Mock()
            mock_redis_client.setex.return_value = None
            mock_redis.return_value = mock_redis_client
            
            redis_service = RedisCacheService(config)
            
            # Test concurrent cache operations
            tasks = []
            for i in range(10):
                task = redis_service.set_cache(f'key_{i}', {'data': f'value_{i}'})
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            assert all(results) == True

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
