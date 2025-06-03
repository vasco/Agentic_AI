from typing import Dict, Any, Optional
from .base_tool import BaseTool
import os
import re
import markdown
import pdfkit
import tempfile
import datetime
from dotenv import load_dotenv

load_dotenv()

class MarkdownToPDF(BaseTool):
    """Tool for converting markdown content to PDF."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the markdown to PDF converter tool.
        
        Args:
            output_dir: Directory where PDF files will be saved (default: "output")
        """
        super().__init__(
            name="markdown_to_pdf",
            description="Convert markdown content to PDF and save it to the specified output directory"
        )
        self.output_dir = output_dir
        
    def _extract_title(self, markdown_text: str) -> str:
        """Extract title from markdown content or generate timestamp-based name."""
        # Try to find the first heading
        heading_match = re.search(r'^#\s+(.+)$', markdown_text, re.MULTILINE)
        if heading_match:
            # Clean the title to be usable as a filename
            title = re.sub(r'[^\w\s-]', '', heading_match.group(1)).strip()
            title = re.sub(r'[-\s]+', '-', title)
            return title.lower()
        else:
            # Use timestamp if no heading found
            return f"document-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def execute(self, markdown_text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert markdown content to PDF and save it to the output directory.
        
        Args:
            markdown_text: The markdown content to convert
            filename: Optional filename for the PDF (without extension)
            
        Returns:
            Dictionary indicating success or failure and the path to the saved PDF file
        """
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Get filename from parameter or extract from content
            if not filename:
                filename = self._extract_title(markdown_text)
            
            # Ensure filename has .pdf extension
            if not filename.endswith('.pdf'):
                filename = f"{filename}.pdf"
            
            output_path = os.path.join(self.output_dir, filename)
                
            # Convert markdown to HTML
            html = markdown.markdown(
                markdown_text,
                extensions=[
                    'markdown.extensions.tables',
                    'markdown.extensions.fenced_code',
                    'markdown.extensions.codehilite',
                    'markdown.extensions.toc'
                ]
            )
            
            # Add CSS for better styling
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    h1, h2, h3, h4, h5, h6 {{ color: #333; margin-top: 24px; margin-bottom: 16px; }}
                    h1 {{ font-size: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #eaecef; }}
                    h2 {{ font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid #eaecef; }}
                    code {{ background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }}
                    pre {{ background-color: #f6f8fa; padding: 16px; overflow: auto; border-radius: 3px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    table, th, td {{ border: 1px solid #ddd; padding: 8px; }}
                    th {{ background-color: #f2f2f2; text-align: left; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    a {{ color: #0366d6; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    ul, ol {{ padding-left: 2em; }}
                    blockquote {{ color: #6a737d; border-left: 0.25em solid #dfe2e5; padding: 0 1em; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
            
            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_html:
                temp_html.write(styled_html.encode('utf-8'))
                temp_html_path = temp_html.name
            
            # Convert HTML to PDF
            pdfkit.from_file(temp_html_path, output_path)
            
            # Clean up temporary file
            os.unlink(temp_html_path)
            
            return {
                "success": True, 
                "file_path": output_path,
                "message": f"PDF successfully created and saved to {output_path}"
            }
            
        except ImportError as e:
            missing_package = str(e).split("'")[1] if "'" in str(e) else "required package"
            return {
                "success": False, 
                "error": f"{missing_package} not installed. Install with 'pip install {missing_package}'"
            }
        except Exception as e:
            return {"success": False, "error": f"PDF conversion failed: {str(e)}"}