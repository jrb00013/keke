import asyncio
import aiohttp
import boto3
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import base64
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
from google.cloud import bigquery
import redis
import pymongo
from sqlalchemy import create_engine, text
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CloudConfig:
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_region: str = 'us-east-1'
    azure_connection_string: Optional[str] = None
    gcp_credentials_path: Optional[str] = None
    redis_url: str = 'redis://localhost:6379'
    mongodb_url: str = 'mongodb://localhost:27017'
    postgres_url: str = 'postgresql://localhost:5432/keke'

class AWSCloudService:
    """AWS cloud services integration"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.session = boto3.Session(
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_secret_key,
            region_name=config.aws_region
        )
        
        # Initialize AWS services
        self.s3 = self.session.client('s3')
        self.lambda_client = self.session.client('lambda')
        self.dynamodb = self.session.resource('dynamodb')
        self.sqs = self.session.client('sqs')
        self.sns = self.session.client('sns')
        self.cloudwatch = self.session.client('cloudwatch')
        
    async def upload_to_s3(self, bucket: str, key: str, data: bytes, 
                          content_type: str = 'application/octet-stream') -> bool:
        """Upload data to S3"""
        try:
            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type
            )
            logger.info(f"Uploaded {key} to S3 bucket {bucket}")
            return True
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return False
    
    async def download_from_s3(self, bucket: str, key: str) -> Optional[bytes]:
        """Download data from S3"""
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"S3 download error: {e}")
            return None
    
    async def invoke_lambda(self, function_name: str, payload: Dict) -> Optional[Dict]:
        """Invoke AWS Lambda function"""
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            result = json.loads(response['Payload'].read())
            return result
        except Exception as e:
            logger.error(f"Lambda invocation error: {e}")
            return None
    
    async def store_in_dynamodb(self, table_name: str, item: Dict) -> bool:
        """Store item in DynamoDB"""
        try:
            table = self.dynamodb.Table(table_name)
            table.put_item(Item=item)
            logger.info(f"Stored item in DynamoDB table {table_name}")
            return True
        except Exception as e:
            logger.error(f"DynamoDB storage error: {e}")
            return False
    
    async def get_from_dynamodb(self, table_name: str, key: Dict) -> Optional[Dict]:
        """Get item from DynamoDB"""
        try:
            table = self.dynamodb.Table(table_name)
            response = table.get_item(Key=key)
            return response.get('Item')
        except Exception as e:
            logger.error(f"DynamoDB retrieval error: {e}")
            return None
    
    async def send_sqs_message(self, queue_url: str, message: str, 
                              delay_seconds: int = 0) -> bool:
        """Send message to SQS queue"""
        try:
            response = self.sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=message,
                DelaySeconds=delay_seconds
            )
            logger.info(f"Sent message to SQS queue {queue_url}")
            return True
        except Exception as e:
            logger.error(f"SQS send error: {e}")
            return False
    
    async def publish_sns_message(self, topic_arn: str, message: str, 
                                 subject: str = None) -> bool:
        """Publish message to SNS topic"""
        try:
            params = {
                'TopicArn': topic_arn,
                'Message': message
            }
            if subject:
                params['Subject'] = subject
                
            self.sns.publish(**params)
            logger.info(f"Published message to SNS topic {topic_arn}")
            return True
        except Exception as e:
            logger.error(f"SNS publish error: {e}")
            return False
    
    async def put_cloudwatch_metric(self, namespace: str, metric_name: str, 
                                  value: float, unit: str = 'Count') -> bool:
        """Put custom metric to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
            logger.info(f"Put metric {metric_name} to CloudWatch")
            return True
        except Exception as e:
            logger.error(f"CloudWatch metric error: {e}")
            return False

class AzureCloudService:
    """Azure cloud services integration"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        if config.azure_connection_string:
            self.blob_client = BlobServiceClient.from_connection_string(
                config.azure_connection_string
            )
        else:
            self.blob_client = None
    
    async def upload_to_blob(self, container: str, blob_name: str, data: bytes) -> bool:
        """Upload data to Azure Blob Storage"""
        if not self.blob_client:
            logger.error("Azure Blob client not initialized")
            return False
        
        try:
            blob_client = self.blob_client.get_blob_client(
                container=container, blob=blob_name
            )
            blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Uploaded {blob_name} to Azure Blob Storage")
            return True
        except Exception as e:
            logger.error(f"Azure Blob upload error: {e}")
            return False
    
    async def download_from_blob(self, container: str, blob_name: str) -> Optional[bytes]:
        """Download data from Azure Blob Storage"""
        if not self.blob_client:
            logger.error("Azure Blob client not initialized")
            return None
        
        try:
            blob_client = self.blob_client.get_blob_client(
                container=container, blob=blob_name
            )
            return blob_client.download_blob().readall()
        except Exception as e:
            logger.error(f"Azure Blob download error: {e}")
            return None

class GCPCloudService:
    """Google Cloud Platform services integration"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        if config.gcp_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.gcp_credentials_path
        
        self.storage_client = gcs.Client()
        self.bigquery_client = bigquery.Client()
    
    async def upload_to_gcs(self, bucket_name: str, blob_name: str, data: bytes) -> bool:
        """Upload data to Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(data)
            logger.info(f"Uploaded {blob_name} to GCS bucket {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"GCS upload error: {e}")
            return False
    
    async def download_from_gcs(self, bucket_name: str, blob_name: str) -> Optional[bytes]:
        """Download data from Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"GCS download error: {e}")
            return None
    
    async def query_bigquery(self, query: str) -> Optional[List[Dict]]:
        """Execute BigQuery query"""
        try:
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"BigQuery query error: {e}")
            return None

class RedisCacheService:
    """Redis caching service"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.redis_client = redis.from_url(config.redis_url)
    
    async def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value with TTL"""
        try:
            serialized_value = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache value"""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def increment_counter(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter"""
        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            return None

class DatabaseService:
    """Database integration service"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.mongo_client = pymongo.MongoClient(config.mongodb_url)
        self.mongo_db = self.mongo_client.keke
        self.postgres_engine = create_engine(config.postgres_url)
    
    async def store_excel_metadata(self, metadata: Dict) -> bool:
        """Store Excel file metadata in MongoDB"""
        try:
            collection = self.mongo_db.excel_files
            result = collection.insert_one(metadata)
            logger.info(f"Stored Excel metadata with ID {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"MongoDB storage error: {e}")
            return False
    
    async def get_excel_metadata(self, file_id: str) -> Optional[Dict]:
        """Get Excel file metadata from MongoDB"""
        try:
            collection = self.mongo_db.excel_files
            return collection.find_one({'_id': file_id})
        except Exception as e:
            logger.error(f"MongoDB retrieval error: {e}")
            return None
    
    async def store_processing_results(self, results: Dict) -> bool:
        """Store processing results in PostgreSQL"""
        try:
            with self.postgres_engine.connect() as conn:
                query = text("""
                    INSERT INTO processing_results 
                    (file_id, operation_type, status, results, created_at)
                    VALUES (:file_id, :operation_type, :status, :results, :created_at)
                """)
                conn.execute(query, {
                    'file_id': results.get('file_id'),
                    'operation_type': results.get('operation_type'),
                    'status': results.get('status'),
                    'results': json.dumps(results.get('data')),
                    'created_at': datetime.utcnow()
                })
                conn.commit()
            logger.info("Stored processing results in PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL storage error: {e}")
            return False
    
    async def get_processing_history(self, file_id: str) -> List[Dict]:
        """Get processing history for a file"""
        try:
            with self.postgres_engine.connect() as conn:
                query = text("""
                    SELECT * FROM processing_results 
                    WHERE file_id = :file_id 
                    ORDER BY created_at DESC
                """)
                result = conn.execute(query, {'file_id': file_id})
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"PostgreSQL query error: {e}")
            return []

class CloudServiceManager:
    """Centralized cloud services manager"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.aws = AWSCloudService(config)
        self.azure = AzureCloudService(config)
        self.gcp = GCPCloudService(config)
        self.redis = RedisCacheService(config)
        self.database = DatabaseService(config)
        
        # Service availability flags
        self.services_available = {
            'aws': bool(config.aws_access_key and config.aws_secret_key),
            'azure': bool(config.azure_connection_string),
            'gcp': bool(config.gcp_credentials_path),
            'redis': True,  # Assume Redis is available
            'database': True  # Assume databases are available
        }
    
    async def backup_excel_file(self, file_data: bytes, file_name: str, 
                              providers: List[str] = None) -> Dict[str, bool]:
        """Backup Excel file to multiple cloud providers"""
        if providers is None:
            providers = ['aws', 'azure', 'gcp']
        
        results = {}
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        key = f"excel_backups/{timestamp}_{file_name}"
        
        for provider in providers:
            if not self.services_available.get(provider, False):
                results[provider] = False
                continue
            
            try:
                if provider == 'aws':
                    results[provider] = await self.aws.upload_to_s3(
                        'keke-excel-backups', key, file_data
                    )
                elif provider == 'azure':
                    results[provider] = await self.azure.upload_to_blob(
                        'excel-backups', key, file_data
                    )
                elif provider == 'gcp':
                    results[provider] = await self.gcp.upload_to_gcs(
                        'keke-excel-backups', key, file_data
                    )
            except Exception as e:
                logger.error(f"Backup to {provider} failed: {e}")
                results[provider] = False
        
        return results
    
    async def cache_analysis_results(self, file_id: str, analysis: Dict) -> bool:
        """Cache analysis results in Redis"""
        cache_key = f"analysis:{file_id}"
        return await self.redis.set_cache(cache_key, analysis, ttl=7200)  # 2 hours
    
    async def get_cached_analysis(self, file_id: str) -> Optional[Dict]:
        """Get cached analysis results"""
        cache_key = f"analysis:{file_id}"
        return await self.redis.get_cache(cache_key)
    
    async def log_processing_event(self, event_type: str, data: Dict) -> bool:
        """Log processing event to cloud services"""
        event = {
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'keke-excel-tool'
        }
        
        # Store in database
        await self.database.store_processing_results({
            'file_id': data.get('file_id'),
            'operation_type': event_type,
            'status': 'completed',
            'data': event
        })
        
        # Send to SNS if AWS is available
        if self.services_available['aws']:
            await self.aws.publish_sns_message(
                'arn:aws:sns:us-east-1:123456789012:keke-events',
                json.dumps(event),
                f'Keke Event: {event_type}'
            )
        
        return True
    
    async def get_service_status(self) -> Dict[str, Dict]:
        """Get status of all cloud services"""
        status = {}
        
        # Test AWS
        if self.services_available['aws']:
            try:
                await self.aws.put_cloudwatch_metric('Keke/Health', 'ServiceCheck', 1)
                status['aws'] = {'available': True, 'status': 'healthy'}
            except:
                status['aws'] = {'available': False, 'status': 'unhealthy'}
        
        # Test Redis
        try:
            await self.redis.set_cache('health_check', 'ok', ttl=60)
            status['redis'] = {'available': True, 'status': 'healthy'}
        except:
            status['redis'] = {'available': False, 'status': 'unhealthy'}
        
        # Test Database
        try:
            await self.database.get_processing_history('test')
            status['database'] = {'available': True, 'status': 'healthy'}
        except:
            status['database'] = {'available': False, 'status': 'unhealthy'}
        
        return status

# Initialize cloud services
cloud_config = CloudConfig(
    aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_region=os.getenv('AWS_REGION', 'us-east-1'),
    azure_connection_string=os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
    gcp_credentials_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
    redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    mongodb_url=os.getenv('MONGODB_URL', 'mongodb://localhost:27017'),
    postgres_url=os.getenv('POSTGRES_URL', 'postgresql://localhost:5432/keke')
)

cloud_manager = CloudServiceManager(cloud_config)
