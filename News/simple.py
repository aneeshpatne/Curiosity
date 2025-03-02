import markdown
import webbrowser
import tempfile

def convert_markdown_to_html(markdown_text):
    """Converts Markdown to a styled HTML email preview."""
    
    # Convert Markdown to Basic HTML
    html_body = markdown.markdown(markdown_text)
    # Apply enhanced CSS styling for a more professional email look
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Preview</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }}
            .email-container {{
                max-width: 650px;
                background: white;
                margin: auto;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
            }}
            h2 {{
                color: #1a73e8;
                text-align: center;
                margin-top: 0;
                margin-bottom: 20px;
                font-size: 22px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }}
            h3 {{
                color: #333;
                font-size: 18px;
                margin-top: 25px;
                margin-bottom: 15px;
                border-left: 4px solid #1a73e8;
                padding-left: 10px;
            }}
            p {{
                color: #555;
                font-size: 16px;
                margin-bottom: 15px;
            }}
            ul {{
                background: #f9fafc;
                padding: 20px 20px 20px 40px;
                border-radius: 8px;
                border-left: 3px solid #dfe1e5;
                margin: 20px 0;
            }}
            li {{
                margin: 10px 0;
                color: #444;
            }}
            .highlight {{
                background: #fff6dd;
                padding: 3px 6px;
                border-radius: 4px;
                font-weight: 500;
                color: #d67d00;
            }}
            a {{
                color: #1a73e8;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .button {{
                display: block;
                width: fit-content;
                margin: 25px auto;
                padding: 12px 25px;
                background: #1a73e8;
                color: white !important;
                text-decoration: none;
                border-radius: 6px;
                font-weight: bold;
                text-align: center;
                transition: background 0.3s ease;
            }}
            .button:hover {{
                background: #0d5bbd;
                text-decoration: none;
            }}
            blockquote {{
                border-left: 4px solid #ddd;
                padding: 10px 15px;
                margin: 20px 0;
                background: #f9f9f9;
                font-style: italic;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: monospace;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                margin: 15px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f8ff;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .footer {{
                text-align: center;
                font-size: 13px;
                color: #777;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <h2>📢 Curiosity Daily News</h2>
            {html_body}
            <p class="footer">This is an automated email. Stay informed!</p>
        </div>
    </body>
    </html>
    """

    return html_content

import tempfile

def preview_markdown_as_email(markdown_text):
    """Creates a temporary HTML file and opens it in a browser."""
    html_content = convert_markdown_to_html(markdown_text)

    # Create a temporary HTML file for previewing (with UTF-8 encoding)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        f.write(html_content)
        temp_file_path = f.name

    # Open the file in a web browser
    webbrowser.open(f"file://{temp_file_path}")


# Example Markdown Input (you can replace this with user input)
markdown_input = """## International Relations

- **US-Ukraine Relations:** Diplomatic tensions arise.
- **Middle East - Gaza Ceasefire:** Israel proposes an extension.

## Humanitarian Crises

- **Bolivia Bus Crash:** Several fatalities reported.
- **India Avalanche:** Multiple casualties and missing persons.

> **Stay updated with global news daily!**
"""

# Generate and preview the HTML
preview_markdown_as_email(markdown_input)
