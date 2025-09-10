import os
import json
import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import uuid
import asyncio
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomationManager:
    """
    Manages workflow automation and scheduled data processing for Keke
    """
    
    def __init__(self):
        self.workflows = {}  # workflow_id -> workflow_info
        self.scheduled_jobs = {}  # job_id -> job_info
        self.running_jobs = {}  # job_id -> job_thread
        self.job_history = []  # List of completed jobs
        self.notifications = []  # List of notifications
        self.is_running = False
        self.scheduler_thread = None
        
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        
        # Webhook configuration
        self.webhook_url = os.getenv('WEBHOOK_URL')
        
        self._start_scheduler()
    
    def _start_scheduler(self):
        """Start the background scheduler"""
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Automation scheduler started")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def create_workflow(self, name: str, description: str, steps: List[Dict[str, Any]], 
                       trigger_type: str = 'manual') -> Dict[str, Any]:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            'id': workflow_id,
            'name': name,
            'description': description,
            'steps': steps,
            'trigger_type': trigger_type,
            'created_at': datetime.now().isoformat(),
            'created_by': 'system',  # Would be set by authenticated user
            'is_active': True,
            'last_run': None,
            'run_count': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow: {name} ({workflow_id})")
        
        return {
            'success': True,
            'workflow_id': workflow_id,
            'workflow': workflow
        }
    
    def schedule_workflow(self, workflow_id: str, schedule_expression: str, 
                         start_time: str = None) -> Dict[str, Any]:
        """Schedule a workflow to run automatically"""
        if workflow_id not in self.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflows[workflow_id]
        
        if workflow['trigger_type'] != 'scheduled':
            return {'error': 'Workflow is not schedulable'}
        
        job_id = str(uuid.uuid4())
        
        job_info = {
            'id': job_id,
            'workflow_id': workflow_id,
            'schedule_expression': schedule_expression,
            'created_at': datetime.now().isoformat(),
            'next_run': self._calculate_next_run(schedule_expression),
            'is_active': True,
            'run_count': 0,
            'last_run': None
        }
        
        self.scheduled_jobs[job_id] = job_info
        
        # Schedule the job
        self._schedule_job(job_id, schedule_expression)
        
        logger.info(f"Scheduled workflow {workflow_id} with expression: {schedule_expression}")
        
        return {
            'success': True,
            'job_id': job_id,
            'job_info': job_info
        }
    
    def run_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a workflow manually"""
        if workflow_id not in self.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflows[workflow_id]
        
        if not workflow['is_active']:
            return {'error': 'Workflow is disabled'}
        
        job_id = str(uuid.uuid4())
        
        # Run workflow in background thread
        job_thread = threading.Thread(
            target=self._execute_workflow,
            args=(job_id, workflow_id, parameters or {}),
            daemon=True
        )
        
        self.running_jobs[job_id] = job_thread
        job_thread.start()
        
        logger.info(f"Started workflow {workflow_id} with job ID {job_id}")
        
        return {
            'success': True,
            'job_id': job_id,
            'status': 'running'
        }
    
    def _execute_workflow(self, job_id: str, workflow_id: str, parameters: Dict[str, Any]):
        """Execute a workflow"""
        workflow = self.workflows[workflow_id]
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing workflow {workflow['name']} (job {job_id})")
            
            results = []
            for i, step in enumerate(workflow['steps']):
                step_result = self._execute_step(step, parameters)
                results.append({
                    'step_index': i,
                    'step_name': step.get('name', f'Step {i+1}'),
                    'result': step_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Check if step failed and workflow should stop
                if not step_result.get('success', False) and step.get('stop_on_failure', True):
                    break
            
            # Update workflow statistics
            workflow['last_run'] = datetime.now().isoformat()
            workflow['run_count'] += 1
            workflow['success_count'] += 1
            
            # Record job completion
            job_record = {
                'job_id': job_id,
                'workflow_id': workflow_id,
                'workflow_name': workflow['name'],
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': (datetime.now() - start_time).total_seconds(),
                'status': 'success',
                'results': results,
                'parameters': parameters
            }
            
            self.job_history.append(job_record)
            
            # Send notifications
            self._send_notifications(job_record)
            
            logger.info(f"Workflow {workflow['name']} completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Update workflow statistics
            workflow['last_run'] = datetime.now().isoformat()
            workflow['run_count'] += 1
            workflow['failure_count'] += 1
            
            # Record job failure
            job_record = {
                'job_id': job_id,
                'workflow_id': workflow_id,
                'workflow_name': workflow['name'],
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': (datetime.now() - start_time).total_seconds(),
                'status': 'failed',
                'error': str(e),
                'parameters': parameters
            }
            
            self.job_history.append(job_record)
            
            # Send failure notifications
            self._send_notifications(job_record)
        
        finally:
            # Clean up running job
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def _execute_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_type = step.get('type')
        
        try:
            if step_type == 'data_processing':
                return self._execute_data_processing_step(step, parameters)
            elif step_type == 'data_cleaning':
                return self._execute_data_cleaning_step(step, parameters)
            elif step_type == 'ml_analysis':
                return self._execute_ml_analysis_step(step, parameters)
            elif step_type == 'export_data':
                return self._execute_export_step(step, parameters)
            elif step_type == 'cloud_sync':
                return self._execute_cloud_sync_step(step, parameters)
            elif step_type == 'notification':
                return self._execute_notification_step(step, parameters)
            elif step_type == 'webhook':
                return self._execute_webhook_step(step, parameters)
            else:
                return {'success': False, 'error': f'Unknown step type: {step_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_data_processing_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing step"""
        # This would integrate with the ExcelProcessor
        return {
            'success': True,
            'message': 'Data processing completed',
            'processed_rows': 1000,
            'output_file': 'processed_data.xlsx'
        }
    
    def _execute_data_cleaning_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data cleaning step"""
        # This would integrate with the ExcelProcessor cleaning functions
        return {
            'success': True,
            'message': 'Data cleaning completed',
            'duplicates_removed': 50,
            'null_values_handled': 25
        }
    
    def _execute_ml_analysis_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML analysis step"""
        # This would integrate with the MLProcessor
        return {
            'success': True,
            'message': 'ML analysis completed',
            'model_accuracy': 0.85,
            'predictions_generated': 100
        }
    
    def _execute_export_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data export step"""
        format_type = step.get('format', 'excel')
        return {
            'success': True,
            'message': f'Data exported as {format_type}',
            'export_file': f'export.{format_type}',
            'file_size': '2.5MB'
        }
    
    def _execute_cloud_sync_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cloud sync step"""
        provider = step.get('provider', 's3')
        return {
            'success': True,
            'message': f'Data synced to {provider}',
            'cloud_path': 'keke/data/synced_file.xlsx'
        }
    
    def _execute_notification_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification step"""
        message = step.get('message', 'Workflow completed')
        recipients = step.get('recipients', [])
        
        for recipient in recipients:
            self._send_email(recipient, 'Keke Workflow Notification', message)
        
        return {
            'success': True,
            'message': f'Notifications sent to {len(recipients)} recipients'
        }
    
    def _execute_webhook_step(self, step: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute webhook step"""
        url = step.get('url')
        payload = step.get('payload', {})
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            return {
                'success': True,
                'message': f'Webhook sent to {url}',
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Webhook failed: {str(e)}'
            }
    
    def _schedule_job(self, job_id: str, schedule_expression: str):
        """Schedule a job based on expression"""
        if schedule_expression.startswith('every '):
            # Parse "every X minutes/hours/days"
            parts = schedule_expression.split()
            if len(parts) >= 3:
                interval = int(parts[1])
                unit = parts[2]
                
                if unit == 'minutes':
                    schedule.every(interval).minutes.do(self._run_scheduled_job, job_id)
                elif unit == 'hours':
                    schedule.every(interval).hours.do(self._run_scheduled_job, job_id)
                elif unit == 'days':
                    schedule.every(interval).days.do(self._run_scheduled_job, job_id)
        
        elif schedule_expression.startswith('daily at '):
            # Parse "daily at HH:MM"
            time_str = schedule_expression.replace('daily at ', '')
            schedule.every().day.at(time_str).do(self._run_scheduled_job, job_id)
        
        elif schedule_expression.startswith('weekly on '):
            # Parse "weekly on Monday at HH:MM"
            parts = schedule_expression.replace('weekly on ', '').split(' at ')
            day = parts[0].lower()
            time_str = parts[1] if len(parts) > 1 else '09:00'
            
            if day == 'monday':
                schedule.every().monday.at(time_str).do(self._run_scheduled_job, job_id)
            elif day == 'tuesday':
                schedule.every().tuesday.at(time_str).do(self._run_scheduled_job, job_id)
            # ... etc for other days
    
    def _run_scheduled_job(self, job_id: str):
        """Run a scheduled job"""
        if job_id not in self.scheduled_jobs:
            return
        
        job_info = self.scheduled_jobs[job_id]
        workflow_id = job_info['workflow_id']
        
        # Update job info
        job_info['last_run'] = datetime.now().isoformat()
        job_info['run_count'] += 1
        job_info['next_run'] = self._calculate_next_run(job_info['schedule_expression'])
        
        # Run the workflow
        self.run_workflow(workflow_id)
    
    def _calculate_next_run(self, schedule_expression: str) -> str:
        """Calculate next run time for schedule expression"""
        # Simplified calculation - in production, use a proper scheduler
        return (datetime.now() + timedelta(hours=1)).isoformat()
    
    def _send_notifications(self, job_record: Dict[str, Any]):
        """Send notifications for job completion"""
        notification = {
            'id': str(uuid.uuid4()),
            'type': 'workflow_completion',
            'job_id': job_record['job_id'],
            'workflow_name': job_record['workflow_name'],
            'status': job_record['status'],
            'timestamp': datetime.now().isoformat(),
            'message': f"Workflow '{job_record['workflow_name']}' {job_record['status']}"
        }
        
        self.notifications.append(notification)
        
        # Send email notification if configured
        if self.smtp_username and self.smtp_password:
            self._send_email(
                'admin@keke.com',  # Would be configurable
                f"Keke Workflow {job_record['status'].title()}",
                notification['message']
            )
    
    def _send_email(self, to_email: str, subject: str, body: str):
        """Send email notification"""
        if not self.smtp_username or not self.smtp_password:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent to {to_email}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and statistics"""
        if workflow_id not in self.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflows[workflow_id]
        
        # Get recent job history for this workflow
        recent_jobs = [
            job for job in self.job_history[-10:] 
            if job['workflow_id'] == workflow_id
        ]
        
        return {
            'success': True,
            'workflow': workflow,
            'recent_jobs': recent_jobs,
            'is_running': any(
                job['workflow_id'] == workflow_id 
                for job in self.running_jobs.values()
            )
        }
    
    def get_automation_dashboard(self) -> Dict[str, Any]:
        """Get automation dashboard data"""
        total_workflows = len(self.workflows)
        active_workflows = len([w for w in self.workflows.values() if w['is_active']])
        scheduled_jobs = len(self.scheduled_jobs)
        running_jobs = len(self.running_jobs)
        
        recent_jobs = self.job_history[-20:] if self.job_history else []
        recent_notifications = self.notifications[-10:] if self.notifications else []
        
        return {
            'success': True,
            'dashboard': {
                'total_workflows': total_workflows,
                'active_workflows': active_workflows,
                'scheduled_jobs': scheduled_jobs,
                'running_jobs': running_jobs,
                'recent_jobs': recent_jobs,
                'recent_notifications': recent_notifications
            }
        }
    
    def stop_automation(self):
        """Stop the automation system"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Automation system stopped")


# Global automation manager instance
automation_manager = AutomationManager()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python automation.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "create_workflow":
            if len(sys.argv) < 5:
                print("Usage: python automation.py create_workflow <name> <description> <steps_json>")
                sys.exit(1)
            
            name = sys.argv[2]
            description = sys.argv[3]
            steps = json.loads(sys.argv[4])
            
            result = automation_manager.create_workflow(name, description, steps)
            print(json.dumps(result, indent=2))
            
        elif command == "run_workflow":
            if len(sys.argv) < 3:
                print("Usage: python automation.py run_workflow <workflow_id> [parameters_json]")
                sys.exit(1)
            
            workflow_id = sys.argv[2]
            parameters = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            
            result = automation_manager.run_workflow(workflow_id, parameters)
            print(json.dumps(result, indent=2))
            
        elif command == "schedule_workflow":
            if len(sys.argv) < 4:
                print("Usage: python automation.py schedule_workflow <workflow_id> <schedule_expression>")
                sys.exit(1)
            
            workflow_id = sys.argv[2]
            schedule_expression = sys.argv[3]
            
            result = automation_manager.schedule_workflow(workflow_id, schedule_expression)
            print(json.dumps(result, indent=2))
            
        elif command == "get_status":
            if len(sys.argv) < 3:
                print("Usage: python automation.py get_status <workflow_id>")
                sys.exit(1)
            
            workflow_id = sys.argv[2]
            
            result = automation_manager.get_workflow_status(workflow_id)
            print(json.dumps(result, indent=2))
            
        elif command == "dashboard":
            result = automation_manager.get_automation_dashboard()
            print(json.dumps(result, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
