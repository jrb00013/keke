import json
import logging
import asyncio
import websockets
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import uuid
from collections import defaultdict
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborationManager:
    """
    Manages real-time collaboration features for Keke
    """
    
    def __init__(self):
        self.active_sessions = {}  # session_id -> session_info
        self.user_sessions = defaultdict(set)  # user_id -> set of session_ids
        self.websocket_connections = {}  # websocket_id -> websocket_info
        self.session_locks = {}  # session_id -> threading.Lock
        self.change_history = defaultdict(list)  # session_id -> list of changes
        self.cursors = defaultdict(dict)  # session_id -> {user_id: cursor_info}
        
    def create_session(self, user_id: str, session_name: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())
        
        session_info = {
            'id': session_id,
            'name': session_name,
            'owner': user_id,
            'created_at': datetime.now().isoformat(),
            'file_info': file_info,
            'participants': {user_id: {
                'joined_at': datetime.now().isoformat(),
                'permissions': 'owner',
                'status': 'active'
            }},
            'changes': [],
            'version': 1
        }
        
        self.active_sessions[session_id] = session_info
        self.user_sessions[user_id].add(session_id)
        self.session_locks[session_id] = threading.Lock()
        
        logger.info(f"Created collaboration session {session_id} for user {user_id}")
        
        return {
            'success': True,
            'session_id': session_id,
            'session_info': session_info
        }
    
    def join_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Join an existing collaboration session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.active_sessions[session_id]
        
        if user_id in session_info['participants']:
            return {'error': 'User already in session'}
        
        session_info['participants'][user_id] = {
            'joined_at': datetime.now().isoformat(),
            'permissions': 'collaborator',
            'status': 'active'
        }
        
        self.user_sessions[user_id].add(session_id)
        
        # Broadcast user joined event
        self._broadcast_to_session(session_id, {
            'type': 'user_joined',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"User {user_id} joined session {session_id}")
        
        return {
            'success': True,
            'session_info': session_info
        }
    
    def leave_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Leave a collaboration session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.active_sessions[session_id]
        
        if user_id not in session_info['participants']:
            return {'error': 'User not in session'}
        
        # Remove user from session
        del session_info['participants'][user_id]
        self.user_sessions[user_id].discard(session_id)
        
        # Clean up user's cursor
        if user_id in self.cursors[session_id]:
            del self.cursors[session_id][user_id]
        
        # Broadcast user left event
        self._broadcast_to_session(session_id, {
            'type': 'user_left',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # If no participants left, close session
        if not session_info['participants']:
            self._close_session(session_id)
        
        logger.info(f"User {user_id} left session {session_id}")
        
        return {'success': True}
    
    def apply_change(self, user_id: str, session_id: str, change: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a change to a collaboration session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.active_sessions[session_id]
        
        if user_id not in session_info['participants']:
            return {'error': 'User not in session'}
        
        # Add change to history
        change_entry = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'change': change,
            'timestamp': datetime.now().isoformat(),
            'version': session_info['version']
        }
        
        with self.session_locks[session_id]:
            session_info['changes'].append(change_entry)
            session_info['version'] += 1
        
        # Broadcast change to all participants
        self._broadcast_to_session(session_id, {
            'type': 'change_applied',
            'change': change_entry,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Applied change {change_entry['id']} to session {session_id}")
        
        return {
            'success': True,
            'change_id': change_entry['id'],
            'version': session_info['version']
        }
    
    def update_cursor(self, user_id: str, session_id: str, cursor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update user's cursor position"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        if user_id not in self.active_sessions[session_id]['participants']:
            return {'error': 'User not in session'}
        
        self.cursors[session_id][user_id] = {
            **cursor_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast cursor update to other participants
        self._broadcast_to_session(session_id, {
            'type': 'cursor_update',
            'user_id': user_id,
            'cursor': self.cursors[session_id][user_id],
            'timestamp': datetime.now().isoformat()
        }, exclude_user=user_id)
        
        return {'success': True}
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current state of a collaboration session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.active_sessions[session_id]
        
        return {
            'success': True,
            'session_info': session_info,
            'cursors': dict(self.cursors[session_id]),
            'change_count': len(session_info['changes'])
        }
    
    def get_user_sessions(self, user_id: str) -> Dict[str, Any]:
        """Get all sessions for a user"""
        user_session_ids = self.user_sessions[user_id]
        sessions = []
        
        for session_id in user_session_ids:
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]
                sessions.append({
                    'id': session_id,
                    'name': session_info['name'],
                    'owner': session_info['owner'],
                    'participant_count': len(session_info['participants']),
                    'created_at': session_info['created_at'],
                    'version': session_info['version']
                })
        
        return {
            'success': True,
            'sessions': sessions
        }
    
    def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: str = None):
        """Broadcast message to all participants in a session"""
        if session_id not in self.active_sessions:
            return
        
        session_info = self.active_sessions[session_id]
        
        for user_id in session_info['participants']:
            if exclude_user and user_id == exclude_user:
                continue
            
            # Find websocket connections for this user
            for ws_id, ws_info in self.websocket_connections.items():
                if ws_info['user_id'] == user_id and ws_info['session_id'] == session_id:
                    try:
                        asyncio.create_task(ws_info['websocket'].send(json.dumps(message)))
                    except Exception as e:
                        logger.error(f"Failed to send message to {user_id}: {e}")
    
    def _close_session(self, session_id: str):
        """Close a collaboration session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.session_locks:
            del self.session_locks[session_id]
        
        if session_id in self.cursors:
            del self.cursors[session_id]
        
        logger.info(f"Closed collaboration session {session_id}")
    
    def register_websocket(self, websocket, user_id: str, session_id: str) -> str:
        """Register a websocket connection"""
        ws_id = str(uuid.uuid4())
        
        self.websocket_connections[ws_id] = {
            'websocket': websocket,
            'user_id': user_id,
            'session_id': session_id,
            'connected_at': datetime.now().isoformat()
        }
        
        logger.info(f"Registered websocket {ws_id} for user {user_id} in session {session_id}")
        
        return ws_id
    
    def unregister_websocket(self, ws_id: str):
        """Unregister a websocket connection"""
        if ws_id in self.websocket_connections:
            del self.websocket_connections[ws_id]
            logger.info(f"Unregistered websocket {ws_id}")


# Global collaboration manager instance
collaboration_manager = CollaborationManager()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python collaboration.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "create_session":
            if len(sys.argv) < 5:
                print("Usage: python collaboration.py create_session <user_id> <session_name> <file_info_json>")
                sys.exit(1)
            
            user_id = sys.argv[2]
            session_name = sys.argv[3]
            file_info = json.loads(sys.argv[4])
            
            result = collaboration_manager.create_session(user_id, session_name, file_info)
            print(json.dumps(result, indent=2))
            
        elif command == "join_session":
            if len(sys.argv) < 4:
                print("Usage: python collaboration.py join_session <user_id> <session_id>")
                sys.exit(1)
            
            user_id = sys.argv[2]
            session_id = sys.argv[3]
            
            result = collaboration_manager.join_session(user_id, session_id)
            print(json.dumps(result, indent=2))
            
        elif command == "leave_session":
            if len(sys.argv) < 4:
                print("Usage: python collaboration.py leave_session <user_id> <session_id>")
                sys.exit(1)
            
            user_id = sys.argv[2]
            session_id = sys.argv[3]
            
            result = collaboration_manager.leave_session(user_id, session_id)
            print(json.dumps(result, indent=2))
            
        elif command == "apply_change":
            if len(sys.argv) < 5:
                print("Usage: python collaboration.py apply_change <user_id> <session_id> <change_json>")
                sys.exit(1)
            
            user_id = sys.argv[2]
            session_id = sys.argv[3]
            change = json.loads(sys.argv[4])
            
            result = collaboration_manager.apply_change(user_id, session_id, change)
            print(json.dumps(result, indent=2))
            
        elif command == "get_session_state":
            if len(sys.argv) < 3:
                print("Usage: python collaboration.py get_session_state <session_id>")
                sys.exit(1)
            
            session_id = sys.argv[2]
            
            result = collaboration_manager.get_session_state(session_id)
            print(json.dumps(result, indent=2))
            
        elif command == "get_user_sessions":
            if len(sys.argv) < 3:
                print("Usage: python collaboration.py get_user_sessions <user_id>")
                sys.exit(1)
            
            user_id = sys.argv[2]
            
            result = collaboration_manager.get_user_sessions(user_id)
            print(json.dumps(result, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
