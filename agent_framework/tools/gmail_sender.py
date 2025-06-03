from typing import Dict, List, Any, Optional, Union
from .base_tool import BaseTool
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os.path
from dotenv import load_dotenv

class GmailSender(BaseTool):
    """Tool for sending emails with attachments via Gmail."""
    
    def __init__(self, email: Optional[str] = None, app_password: Optional[str] = None):
        """
        Initialize the Gmail sender tool.
        
        Args:
            email: Gmail email address
            app_password: App password for Gmail (generate from Google Account settings)
        """
        super().__init__(
            name="gmail_sender",
            description="Send emails with optional attachments via Gmail"
        )
        load_dotenv()
        self.email = email or os.environ.get("GMAIL_EMAIL")
        self.app_password = app_password or os.environ.get("GMAIL_APP_PASSWORD")
    
    def execute(
        self, 
        to: Union[str, List[str]], 
        subject: str, 
        body: str, 
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send an email via Gmail with optional attachments.
        
        Args:
            to: Recipient email address(es) - can be a string or list of strings
            subject: Email subject
            body: Email body text
            attachments: Optional list of file paths to attach to the email
            
        Returns:
            Dictionary with success status and details
        """
        try:
            if not self.email or not self.app_password:
                return {
                    "success": False, 
                    "error": "Gmail credentials are required. Please provide email and app password."
                }
            
            # Create message container
            msg = MIMEMultipart()
            msg['From'] = self.email
            
            # Handle different types of recipients
            if isinstance(to, list):
                msg['To'] = ', '.join(to)
                recipients = to
            else:
                msg['To'] = to
                recipients = [to]
                
            msg['Subject'] = subject
            
            # Attach the body text
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach files if provided
            if attachments:
                for file_path in attachments:
                    if not os.path.exists(file_path):
                        return {
                            "success": False, 
                            "error": f"Attachment file not found: {file_path}"
                        }
                    
                    with open(file_path, 'rb') as file:
                        attachment = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                    
                    # Add header with filename
                    attachment['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                    msg.attach(attachment)
            
            # Connect to Gmail server and send
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email, self.app_password)
            server.sendmail(self.email, recipients, msg.as_string())
            server.quit()
            
            return {
                "success": True, 
                "message": f"Email successfully sent to {', '.join(recipients)}"
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                "success": False, 
                "error": "Authentication failed. Please check your email and app password."
            }
        except smtplib.SMTPException as e:
            return {
                "success": False, 
                "error": f"SMTP error occurred: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Failed to send email: {str(e)}"
            }