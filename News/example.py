import smtplib
from email.mime.text import MIMEText

# iCloud SMTP settings
SMTP_SERVER = "smtp.mail.me.com"
SMTP_PORT = 587  # Use 465 if SSL
EMAIL = "libra_anil@icloud.com"
APP_PASSWORD = "ynpw-ptkg-cgzq-xugg"  # Use the generated app-specific password

# Email content
msg = MIMEText("This is a test email from iCloud SMTP.")
msg["Subject"] = "Test Email"
msg["From"] = "curiosity@aneeshpatne.com"
msg["To"] = "aneeshpatne@gmail.com"

# Send email
try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL, APP_PASSWORD)
    server.sendmail(EMAIL, ["aneeshpatne@gmail.com"], msg.as_string())
    server.quit()
    print("✅ Email sent successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
