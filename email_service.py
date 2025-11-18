"""
Email Service - Send notifications and reports via email
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Dict, List, Optional
import json


class EmailService:
    """Email notification service"""

    def __init__(self, db):
        self.db = db
        self.smtp_config = self._load_config()

    def _load_config(self) -> Dict:
        """Load email configuration from database"""
        config = self.db.get_setting('email_config')
        if config:
            return json.loads(config)
        return {
            'enabled': False,
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'username': '',
            'password': '',
            'from_email': '',
            'from_name': 'AI Gallery'
        }

    def save_config(self, config: Dict) -> bool:
        """Save email configuration"""
        self.smtp_config = config
        self.db.set_setting('email_config', json.dumps(config))
        return True

    def test_connection(self) -> Dict:
        """Test SMTP connection"""
        if not self.smtp_config.get('enabled'):
            return {'success': False, 'error': 'Email not configured'}

        try:
            server = self._create_smtp_connection()
            server.quit()
            return {'success': True, 'message': 'Connection successful'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _create_smtp_connection(self):
        """Create SMTP connection"""
        config = self.smtp_config

        if config.get('use_tls'):
            server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(config['smtp_host'], config['smtp_port'])

        if config.get('username') and config.get('password'):
            server.login(config['username'], config['password'])

        return server

    def send_email(self, to_email: str, subject: str, body: str,
                   html_body: Optional[str] = None,
                   attachments: Optional[List[str]] = None) -> Dict:
        """Send an email"""
        if not self.smtp_config.get('enabled'):
            return {'success': False, 'error': 'Email service not enabled'}

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.smtp_config['from_name']} <{self.smtp_config['from_email']}>"
            msg['To'] = to_email

            # Add text body
            msg.attach(MIMEText(body, 'plain'))

            # Add HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Add attachments
            if attachments:
                for filepath in attachments:
                    self._attach_file(msg, filepath)

            # Send email
            server = self._create_smtp_connection()
            server.send_message(msg)
            server.quit()

            return {'success': True, 'message': 'Email sent successfully'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _attach_file(self, msg: MIMEMultipart, filepath: str):
        """Attach a file to email"""
        path = Path(filepath)

        if not path.exists():
            return

        # Check if it's an image
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            with open(filepath, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=path.name)
                msg.attach(img)
        else:
            # Generic attachment
            with open(filepath, 'rb') as f:
                from email.mime.base import MIMEBase
                from email import encoders

                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={path.name}')
                msg.attach(part)

    def send_pipeline_report(self, pipeline_name: str, execution_result: Dict,
                            to_email: str) -> Dict:
        """Send pipeline execution report"""
        subject = f"Pipeline Completed: {pipeline_name}"

        # Plain text body
        body = f"""
Pipeline Execution Report
========================

Pipeline: {pipeline_name}
Status: {execution_result.get('status', 'unknown')}
Total Images: {execution_result.get('total_images', 0)}
Successful: {execution_result.get('successful', 0)}
Failed: {execution_result.get('failed', 0)}

Execution ID: {execution_result.get('execution_id')}

This is an automated notification from AI Gallery.
"""

        # HTML body
        html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .stat {{ margin: 10px 0; }}
        .success {{ color: #4CAF50; }}
        .error {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🚀 Pipeline Execution Report</h2>
    </div>
    <div class="content">
        <h3>{pipeline_name}</h3>
        <div class="stats">
            <div class="stat"><strong>Status:</strong> {execution_result.get('status', 'unknown')}</div>
            <div class="stat"><strong>Total Images:</strong> {execution_result.get('total_images', 0)}</div>
            <div class="stat success"><strong>Successful:</strong> {execution_result.get('successful', 0)}</div>
            <div class="stat error"><strong>Failed:</strong> {execution_result.get('failed', 0)}</div>
            <div class="stat"><strong>Execution ID:</strong> {execution_result.get('execution_id')}</div>
        </div>
        <p style="margin-top: 20px; color: #666;">
            This is an automated notification from AI Gallery.
        </p>
    </div>
</body>
</html>
"""

        return self.send_email(to_email, subject, body, html_body)

    def send_image_notification(self, image: Dict, event: str, to_email: str,
                               attach_image: bool = False) -> Dict:
        """Send notification about an image"""
        subject = f"Image {event}: {image.get('filename', 'Unknown')}"

        body = f"""
Image Notification
=================

Event: {event}
Filename: {image.get('filename', 'Unknown')}
Path: {image.get('filepath', 'Unknown')}
Tags: {', '.join(json.loads(image.get('tags', '[]')) if image.get('tags') else [])}

This is an automated notification from AI Gallery.
"""

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background: #2196F3; color: white; padding: 20px;">
        <h2>📷 Image Notification</h2>
    </div>
    <div style="padding: 20px;">
        <h3>{event}</h3>
        <p><strong>Filename:</strong> {image.get('filename', 'Unknown')}</p>
        <p><strong>Path:</strong> {image.get('filepath', 'Unknown')}</p>
        <p><strong>Tags:</strong> {', '.join(json.loads(image.get('tags', '[]')) if image.get('tags') else [])}</p>
    </div>
</body>
</html>
"""

        attachments = []
        if attach_image and image.get('filepath'):
            attachments.append(image['filepath'])

        return self.send_email(to_email, subject, body, html_body, attachments)


# Global instance
_email_service = None


def get_email_service(db=None):
    """Get or create email service singleton"""
    global _email_service

    if _email_service is None and db:
        _email_service = EmailService(db)

    return _email_service
