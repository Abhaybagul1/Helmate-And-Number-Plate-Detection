import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.message import EmailMessage
import os


def send_email_alert(plate_number, image_path):
    sender_email = "abhaybagul2003@gmail.com"
    receiver_email = "abhaybagul360@gmail.com"
    password = "zixs rdco fbkd rcva"  # ❌ Do NOT hardcode passwords (use environment variables instead)

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = f"Helmet Violation Detected - {plate_number}"

    body = f"A violation has been detected. The vehicle number is {plate_number}. See the attached image."
    msg.attach(MIMEText(body, "plain"))
    

    # Attach the image
    if os.path.exists(image_path):  # Check if image exists
        with open(image_path, "rb") as img:
            attachment = MIMEBase("application", "octet-stream")
            attachment.set_payload(img.read())
            encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(attachment)
    else:
        print(f"Error: Image not found at {image_path}")

    try:
        # Connect to Gmail SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("✅ Email sent successfully!")

    except smtplib.SMTPAuthenticationError:
        print("❌ Authentication Error: Check your email and password!")
    except Exception as e:
        print(f"❌ Error: {e}")

# Test the function
send_email_alert("MH12AB1234", "C:/project1/myproject/train/images/new42.jpg")



