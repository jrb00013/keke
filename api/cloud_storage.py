import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudStorageManager:
    """
    Manages cloud storage integrations for Keke
    """
    
    def __init__(self):
        self.s3_client = None
        self.s3_bucket = os.getenv('AWS_S3_BUCKET')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Google Drive credentials
        self.google_drive_token = os.getenv('GOOGLE_DRIVE_TOKEN')
        self.google_drive_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        
        # Dropbox credentials
        self.dropbox_token = os.getenv('DROPBOX_TOKEN')
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize cloud storage clients"""
        try:
            # Initialize S3 client
            if self.aws_access_key and self.aws_secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
    
    def upload_to_s3(self, file_path: str, key: str, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Upload file to AWS S3
        """
        if not self.s3_client or not self.s3_bucket:
            return {'error': 'S3 not configured'}
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(file_path, self.s3_bucket, key, ExtraArgs=extra_args)
            
            # Generate presigned URL for download
            download_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': key},
                ExpiresIn=3600  # 1 hour
            )
            
            return {
                'success': True,
                'bucket': self.s3_bucket,
                'key': key,
                'download_url': download_url,
                'upload_time': datetime.now().isoformat()
            }
            
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            return {'error': f'S3 upload failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            return {'error': f'Upload failed: {str(e)}'}
    
    def download_from_s3(self, key: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from AWS S3
        """
        if not self.s3_client or not self.s3_bucket:
            return {'error': 'S3 not configured'}
        
        try:
            self.s3_client.download_file(self.s3_bucket, key, local_path)
            
            return {
                'success': True,
                'local_path': local_path,
                'key': key,
                'download_time': datetime.now().isoformat()
            }
            
        except ClientError as e:
            logger.error(f"S3 download error: {e}")
            return {'error': f'S3 download failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error during S3 download: {e}")
            return {'error': f'Download failed: {str(e)}'}
    
    def list_s3_files(self, prefix: str = '') -> Dict[str, Any]:
        """
        List files in S3 bucket
        """
        if not self.s3_client or not self.s3_bucket:
            return {'error': 'S3 not configured'}
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag']
                    })
            
            return {
                'success': True,
                'files': files,
                'count': len(files)
            }
            
        except ClientError as e:
            logger.error(f"S3 list error: {e}")
            return {'error': f'S3 list failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error during S3 list: {e}")
            return {'error': f'List failed: {str(e)}'}
    
    def upload_to_google_drive(self, file_path: str, name: str, description: str = '') -> Dict[str, Any]:
        """
        Upload file to Google Drive
        """
        if not self.google_drive_token:
            return {'error': 'Google Drive not configured'}
        
        try:
            # Google Drive API endpoint
            url = 'https://www.googleapis.com/upload/drive/v3/files'
            
            headers = {
                'Authorization': f'Bearer {self.google_drive_token}'
            }
            
            # File metadata
            metadata = {
                'name': name,
                'description': description
            }
            
            if self.google_drive_folder_id:
                metadata['parents'] = [self.google_drive_folder_id]
            
            # Upload file
            with open(file_path, 'rb') as file_data:
                files = {'file': file_data}
                data = {'metadata': json.dumps(metadata)}
                
                response = requests.post(
                    f"{url}?uploadType=multipart",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'file_id': result['id'],
                    'name': result['name'],
                    'web_view_link': result.get('webViewLink'),
                    'upload_time': datetime.now().isoformat()
                }
            else:
                return {'error': f'Google Drive upload failed: {response.text}'}
                
        except Exception as e:
            logger.error(f"Google Drive upload error: {e}")
            return {'error': f'Google Drive upload failed: {str(e)}'}
    
    def download_from_google_drive(self, file_id: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from Google Drive
        """
        if not self.google_drive_token:
            return {'error': 'Google Drive not configured'}
        
        try:
            # Get file metadata first
            metadata_url = f'https://www.googleapis.com/drive/v3/files/{file_id}'
            headers = {'Authorization': f'Bearer {self.google_drive_token}'}
            
            metadata_response = requests.get(metadata_url, headers=headers)
            if metadata_response.status_code != 200:
                return {'error': 'Failed to get file metadata'}
            
            file_metadata = metadata_response.json()
            
            # Download file content
            download_url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'
            
            response = requests.get(download_url, headers=headers)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                return {
                    'success': True,
                    'local_path': local_path,
                    'file_id': file_id,
                    'name': file_metadata['name'],
                    'download_time': datetime.now().isoformat()
                }
            else:
                return {'error': f'Google Drive download failed: {response.text}'}
                
        except Exception as e:
            logger.error(f"Google Drive download error: {e}")
            return {'error': f'Google Drive download failed: {str(e)}'}
    
    def upload_to_dropbox(self, file_path: str, dropbox_path: str) -> Dict[str, Any]:
        """
        Upload file to Dropbox
        """
        if not self.dropbox_token:
            return {'error': 'Dropbox not configured'}
        
        try:
            url = 'https://content.dropboxapi.com/2/files/upload'
            
            headers = {
                'Authorization': f'Bearer {self.dropbox_token}',
                'Dropbox-API-Arg': json.dumps({
                    'path': dropbox_path,
                    'mode': 'add',
                    'autorename': True
                }),
                'Content-Type': 'application/octet-stream'
            }
            
            with open(file_path, 'rb') as file_data:
                response = requests.post(url, headers=headers, data=file_data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'path': result['path_display'],
                    'id': result['id'],
                    'upload_time': datetime.now().isoformat()
                }
            else:
                return {'error': f'Dropbox upload failed: {response.text}'}
                
        except Exception as e:
            logger.error(f"Dropbox upload error: {e}")
            return {'error': f'Dropbox upload failed: {str(e)}'}
    
    def download_from_dropbox(self, dropbox_path: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from Dropbox
        """
        if not self.dropbox_token:
            return {'error': 'Dropbox not configured'}
        
        try:
            url = 'https://content.dropboxapi.com/2/files/download'
            
            headers = {
                'Authorization': f'Bearer {self.dropbox_token}',
                'Dropbox-API-Arg': json.dumps({'path': dropbox_path})
            }
            
            response = requests.post(url, headers=headers)
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                return {
                    'success': True,
                    'local_path': local_path,
                    'dropbox_path': dropbox_path,
                    'download_time': datetime.now().isoformat()
                }
            else:
                return {'error': f'Dropbox download failed: {response.text}'}
                
        except Exception as e:
            logger.error(f"Dropbox download error: {e}")
            return {'error': f'Dropbox download failed: {str(e)}'}
    
    def get_cloud_storage_status(self) -> Dict[str, Any]:
        """
        Get status of all cloud storage integrations
        """
        status = {
            'aws_s3': {
                'configured': bool(self.s3_client and self.s3_bucket),
                'bucket': self.s3_bucket,
                'region': self.aws_region
            },
            'google_drive': {
                'configured': bool(self.google_drive_token),
                'folder_id': self.google_drive_folder_id
            },
            'dropbox': {
                'configured': bool(self.dropbox_token)
            }
        }
        
        return status
    
    def sync_file_to_cloud(self, file_path: str, cloud_provider: str, 
                          cloud_path: str = None, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Sync a file to specified cloud provider
        """
        if not os.path.exists(file_path):
            return {'error': 'File does not exist'}
        
        file_name = os.path.basename(file_path)
        
        if cloud_provider.lower() == 's3':
            key = cloud_path or f'keke/{file_name}'
            return self.upload_to_s3(file_path, key, metadata)
        
        elif cloud_provider.lower() == 'google_drive':
            return self.upload_to_google_drive(file_path, file_name, metadata.get('description', ''))
        
        elif cloud_provider.lower() == 'dropbox':
            db_path = cloud_path or f'/keke/{file_name}'
            return self.upload_to_dropbox(file_path, db_path)
        
        else:
            return {'error': f'Unsupported cloud provider: {cloud_provider}'}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cloud_storage.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    manager = CloudStorageManager()
    
    try:
        if command == "status":
            result = manager.get_cloud_storage_status()
            print(json.dumps(result, indent=2))
            
        elif command == "upload":
            if len(sys.argv) < 5:
                print("Usage: python cloud_storage.py upload <file_path> <provider> <cloud_path>")
                sys.exit(1)
            
            file_path = sys.argv[2]
            provider = sys.argv[3]
            cloud_path = sys.argv[4]
            
            result = manager.sync_file_to_cloud(file_path, provider, cloud_path)
            print(json.dumps(result, indent=2))
            
        elif command == "download":
            if len(sys.argv) < 5:
                print("Usage: python cloud_storage.py download <provider> <cloud_path> <local_path>")
                sys.exit(1)
            
            provider = sys.argv[2]
            cloud_path = sys.argv[3]
            local_path = sys.argv[4]
            
            if provider.lower() == 's3':
                result = manager.download_from_s3(cloud_path, local_path)
            elif provider.lower() == 'google_drive':
                result = manager.download_from_google_drive(cloud_path, local_path)
            elif provider.lower() == 'dropbox':
                result = manager.download_from_dropbox(cloud_path, local_path)
            else:
                result = {'error': f'Unsupported provider: {provider}'}
            
            print(json.dumps(result, indent=2))
            
        elif command == "list":
            if len(sys.argv) < 3:
                print("Usage: python cloud_storage.py list <provider>")
                sys.exit(1)
            
            provider = sys.argv[2]
            
            if provider.lower() == 's3':
                prefix = sys.argv[3] if len(sys.argv) > 3 else ''
                result = manager.list_s3_files(prefix)
            else:
                result = {'error': f'List not supported for provider: {provider}'}
            
            print(json.dumps(result, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
