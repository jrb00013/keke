import os
import json
import logging
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Manages security features for Keke including authentication, authorization, and encryption
    """
    
    def __init__(self):
        self.secret_key = os.getenv('KEKE_SECRET_KEY', self._generate_secret_key())
        self.jwt_secret = os.getenv('JWT_SECRET', self.secret_key)
        self.encryption_key = self._derive_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # In-memory user store (in production, use a database)
        self.users = {}
        self.sessions = {}
        self.api_keys = {}
        
        # Rate limiting
        self.rate_limits = {}
        self.max_requests_per_minute = 100
        
        # Initialize default admin user
        self._create_default_admin()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret"""
        password = self.secret_key.encode()
        salt = b'keke_salt_2024'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
        self.create_user('admin', admin_password, 'admin@keke.com', 'admin')
        logger.info("Default admin user created")
    
    def create_user(self, username: str, password: str, email: str, role: str = 'user') -> Dict[str, Any]:
        """Create a new user"""
        if username in self.users:
            return {'error': 'Username already exists'}
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        user = {
            'id': secrets.token_urlsafe(16),
            'username': username,
            'email': email,
            'password_hash': password_hash.decode('utf-8'),
            'role': role,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True,
            'permissions': self._get_role_permissions(role)
        }
        
        self.users[username] = user
        
        logger.info(f"Created user: {username}")
        
        return {
            'success': True,
            'user_id': user['id'],
            'username': username,
            'role': role
        }
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user"""
        if username not in self.users:
            return {'error': 'Invalid credentials'}
        
        user = self.users[username]
        
        if not user['is_active']:
            return {'error': 'Account is disabled'}
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return {'error': 'Invalid credentials'}
        
        # Update last login
        user['last_login'] = datetime.now().isoformat()
        
        # Generate JWT token
        token = self._generate_jwt_token(user)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user['id'],
            'username': username,
            'created_at': datetime.now().isoformat(),
            'expires_at': datetime.now() + timedelta(hours=24),
            'ip_address': None,  # Would be set by the web server
            'user_agent': None
        }
        
        logger.info(f"User {username} authenticated successfully")
        
        return {
            'success': True,
            'token': token,
            'session_id': session_id,
            'user': {
                'id': user['id'],
                'username': username,
                'email': user['email'],
                'role': user['role'],
                'permissions': user['permissions']
            }
        }
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if user still exists and is active
            username = payload.get('username')
            if username not in self.users:
                return {'error': 'User not found'}
            
            user = self.users[username]
            if not user['is_active']:
                return {'error': 'Account is disabled'}
            
            return {
                'success': True,
                'user': {
                    'id': user['id'],
                    'username': username,
                    'email': user['email'],
                    'role': user['role'],
                    'permissions': user['permissions']
                }
            }
            
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}
    
    def check_permission(self, username: str, permission: str) -> bool:
        """Check if user has specific permission"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        return permission in user['permissions']
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return encrypted_data
    
    def generate_api_key(self, username: str, name: str) -> Dict[str, Any]:
        """Generate API key for user"""
        if username not in self.users:
            return {'error': 'User not found'}
        
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            'username': username,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'is_active': True
        }
        
        logger.info(f"Generated API key for user {username}")
        
        return {
            'success': True,
            'api_key': api_key,
            'name': name
        }
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            return {'error': 'Invalid API key'}
        
        key_info = self.api_keys[key_hash]
        
        if not key_info['is_active']:
            return {'error': 'API key is disabled'}
        
        # Update last used
        key_info['last_used'] = datetime.now().isoformat()
        
        return {
            'success': True,
            'username': key_info['username'],
            'name': key_info['name']
        }
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit"""
        now = datetime.now()
        minute_key = now.strftime('%Y-%m-%d-%H-%M')
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {}
        
        if minute_key not in self.rate_limits[identifier]:
            self.rate_limits[identifier][minute_key] = 0
        
        self.rate_limits[identifier][minute_key] += 1
        
        # Clean old entries
        for key in list(self.rate_limits[identifier].keys()):
            if key < minute_key:
                del self.rate_limits[identifier][key]
        
        return self.rate_limits[identifier][minute_key] <= self.max_requests_per_minute
    
    def audit_log(self, username: str, action: str, details: Dict[str, Any] = None):
        """Log security events"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'action': action,
            'details': details or {},
            'ip_address': None  # Would be set by web server
        }
        
        logger.info(f"Security audit: {json.dumps(log_entry)}")
    
    def _generate_jwt_token(self, user: Dict[str, Any]) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user['id'],
            'username': user['username'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for role"""
        permissions = {
            'admin': [
                'read', 'write', 'delete', 'admin', 'manage_users',
                'view_analytics', 'export_data', 'import_data'
            ],
            'user': [
                'read', 'write', 'export_data', 'import_data'
            ],
            'viewer': [
                'read', 'export_data'
            ]
        }
        
        return permissions.get(role, ['read'])
    
    def get_user_info(self, username: str) -> Dict[str, Any]:
        """Get user information"""
        if username not in self.users:
            return {'error': 'User not found'}
        
        user = self.users[username]
        
        return {
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role'],
                'created_at': user['created_at'],
                'last_login': user['last_login'],
                'is_active': user['is_active']
            }
        }
    
    def update_user_role(self, username: str, new_role: str) -> Dict[str, Any]:
        """Update user role"""
        if username not in self.users:
            return {'error': 'User not found'}
        
        user = self.users[username]
        user['role'] = new_role
        user['permissions'] = self._get_role_permissions(new_role)
        
        logger.info(f"Updated role for user {username} to {new_role}")
        
        return {'success': True}
    
    def deactivate_user(self, username: str) -> Dict[str, Any]:
        """Deactivate user account"""
        if username not in self.users:
            return {'error': 'User not found'}
        
        user = self.users[username]
        user['is_active'] = False
        
        logger.info(f"Deactivated user {username}")
        
        return {'success': True}


# Global security manager instance
security_manager = SecurityManager()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python security.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "create_user":
            if len(sys.argv) < 5:
                print("Usage: python security.py create_user <username> <password> <email> [role]")
                sys.exit(1)
            
            username = sys.argv[2]
            password = sys.argv[3]
            email = sys.argv[4]
            role = sys.argv[5] if len(sys.argv) > 5 else 'user'
            
            result = security_manager.create_user(username, password, email, role)
            print(json.dumps(result, indent=2))
            
        elif command == "authenticate":
            if len(sys.argv) < 4:
                print("Usage: python security.py authenticate <username> <password>")
                sys.exit(1)
            
            username = sys.argv[2]
            password = sys.argv[3]
            
            result = security_manager.authenticate_user(username, password)
            print(json.dumps(result, indent=2))
            
        elif command == "verify_token":
            if len(sys.argv) < 3:
                print("Usage: python security.py verify_token <token>")
                sys.exit(1)
            
            token = sys.argv[2]
            
            result = security_manager.verify_token(token)
            print(json.dumps(result, indent=2))
            
        elif command == "generate_api_key":
            if len(sys.argv) < 4:
                print("Usage: python security.py generate_api_key <username> <key_name>")
                sys.exit(1)
            
            username = sys.argv[2]
            key_name = sys.argv[3]
            
            result = security_manager.generate_api_key(username, key_name)
            print(json.dumps(result, indent=2))
            
        elif command == "encrypt":
            if len(sys.argv) < 3:
                print("Usage: python security.py encrypt <data>")
                sys.exit(1)
            
            data = sys.argv[2]
            
            result = security_manager.encrypt_data(data)
            print(json.dumps({'encrypted': result}, indent=2))
            
        elif command == "decrypt":
            if len(sys.argv) < 3:
                print("Usage: python security.py decrypt <encrypted_data>")
                sys.exit(1)
            
            encrypted_data = sys.argv[2]
            
            result = security_manager.decrypt_data(encrypted_data)
            print(json.dumps({'decrypted': result}, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
