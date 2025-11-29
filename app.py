import os
import re
import csv
import logging
import shutil
from datetime import datetime
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import pytesseract

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------- CONFIG --------------------
# Update this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# SMTP CONFIGURATION
SMTP_EMAIL = "harijawaharh2@gmail.com"  # REPLACE THIS
SMTP_PASSWORD = "cliw jusn mghj ygcc"  # REPLACE THIS
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ARCHIVE_FOLDER"] = "archive"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["ARCHIVE_FOLDER"], exist_ok=True)

DATA_FILE = "data.csv"
CONTACT_FILE = "contact.csv"  # Expected: Name,phno,rollno,email
LAST_RUN_FILE = "last_run_date.txt"

FIELDNAMES = [
    "S.No", "Label ID", "Roll No", "Name", "Company", "AWB No",
    "Email", "Phone No", "Time", "Parcel No", "Picked", "Signature",
    "Status", "Mail Status", "Mail Time"
]

logging.basicConfig(level=logging.INFO)

# -------------------- EMAIL UTILS --------------------
def send_email_real(to_email, subject, body):
    if "your_email" in SMTP_EMAIL:
        return False, "SMTP credentials not set."
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL.strip(), SMTP_PASSWORD.replace(" ", "").strip())
        text = msg.as_string()
        server.sendmail(SMTP_EMAIL, to_email, text)
        server.quit()
        logging.info(f"Email sent to {to_email}")
        return True, None
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Failed to send email to {to_email}: {error_msg}")
        return False, error_msg

# -------------------- OCR + CLASSIFICATION --------------------

# -------------------- OCR + CLASSIFICATION --------------------
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

def ocr_extract(path):
    gray = preprocess_image(path)
    text = pytesseract.image_to_string(gray)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return text, lines

def classify_lines(lines):
    name = company = phone = awb = rollno = None
    cleaned = [re.sub(r'[^a-zA-Z0-9\s,+-.]', '', l).strip() for l in lines if len(l.strip()) > 2]

    for line in cleaned:
        # AWB
        if not awb and re.search(r'\b\d{10,15}\b', line):
            awb = re.findall(r'\b\d{10,15}\b', line)[0]
            continue
        # PHONE
        if not phone and re.search(r'(\+91|91)?\s?\d{10}\b', line):
            phone = re.findall(r'\d{10}\b', line)[0]
            continue
        # ROLL NO (Heuristic: 10 chars, starts with digit, e.g., 21691A3155)
        if not rollno and re.search(r'\b\d{2}[A-Z0-9]{8}\b', line, re.IGNORECASE):
            rollno = re.findall(r'\b\d{2}[A-Z0-9]{8}\b', line, re.IGNORECASE)[0]
            continue
        # COMPANY
        if not company and any(x in line.lower() for x in
                               ["flipkart", "ekart", "delhivery", "amazon",
                                "bluedart", "xpressbees", "ecom", "shadowfax"]):
            company = line
            continue
        # NAME
        if not name and re.match(r'^[A-Za-z][A-Za-z\s.]{2,30}$', line):
            name = line.strip()

    return {
        "Name": name or "Not Found",
        "Company": company or "Not Found",
        "Phone No": phone or "Not Found",
        "AWB No": awb or "Not Found",
        "Roll No": rollno or ""
    }

# -------------------- CSV HELPERS --------------------
def read_csv(file_path=DATA_FILE):
    if not os.path.exists(file_path):
        return []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            nr = {k: r.get(k, "") for k in FIELDNAMES}
            rows.append(nr)
        return rows

def write_csv(rows, file_path=DATA_FILE):
    with open(file_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        normalized = []
        for r in rows:
            nr = {k: r.get(k, "") for k in FIELDNAMES}
            normalized.append(nr)
        writer.writerows(normalized)

def get_recipient_details(query_val):
    """
    Lookup recipient details by Roll No or Phone No in CONTACT_FILE.
    """
    if not query_val:
        return None
    if not os.path.exists(CONTACT_FILE):
        return None
    
    query_val = str(query_val).strip().lower()
    
    with open(CONTACT_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_roll = row.get("rollno", "").strip().lower()
            r_phone = row.get("phno", "").strip().lower()
            
            # Check for match
            if (r_roll and query_val == r_roll) or (r_phone and query_val in r_phone):
                 return {
                     "Name": row.get("Name", ""),
                     "Email": row.get("email", ""),
                     "Roll No": row.get("rollno", ""),
                     "Phone No": row.get("phno", "")
                 }
    return None

def check_daily_reset():
    today_str = datetime.now().strftime("%Y-%m-%d")
    last_run = ""
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, "r") as f:
            last_run = f.read().strip()
            
    if last_run and last_run != today_str:
        if os.path.exists(DATA_FILE):
            rows = read_csv()
            if rows:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"data_{last_run}_{timestamp}.csv"
                archive_path = os.path.join(app.config["ARCHIVE_FOLDER"], archive_name)
                shutil.copy(DATA_FILE, archive_path)
                logging.info(f"Archived data to {archive_path}")
                
                # Clear data file
                with open(DATA_FILE, "w", newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
        
    with open(LAST_RUN_FILE, "w") as f:
        f.write(today_str)

def append_to_csv(data):
    check_daily_reset()
    rows = read_csv()
    
    # Label ID: YYYYMMDD-Serial
    today_compact = datetime.now().strftime("%Y%m%d")
    serial = len(rows) + 1
    label_id = f"{today_compact}-{serial:04d}"
    
    s_no = str(len(rows) + 1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Match Recipient
    match = None
    if data.get("Roll No"):
        match = get_recipient_details(data["Roll No"])
    if not match and data.get("Phone No"):
        match = get_recipient_details(data["Phone No"])
        
    final_name = data.get("Name", "")
    final_email = data.get("Email", "")
    final_phone = data.get("Phone No", "")
    final_roll = data.get("Roll No", "")
    
    if match:
        final_name = match["Name"] or final_name
        final_email = match["Email"] or final_email
        final_phone = match["Phone No"] or final_phone
        final_roll = match["Roll No"] or final_roll

    entry = {
        "S.No": s_no,
        "Label ID": label_id,
        "Roll No": final_roll,
        "Name": final_name,
        "Company": data.get("Company", ""),
        "AWB No": data.get("AWB No", ""),
        "Email": final_email,
        "Phone No": final_phone,
        "Time": now,
        "Parcel No": data.get("Parcel No", ""),
        "Picked": data.get("Picked", "Not Picked"),
        "Signature": data.get("Signature", ""),
        "Status": data.get("Status", "Pending"),
        "Mail Status": data.get("Mail Status", "Pending"),
        "Mail Time": data.get("Mail Time", "")
    }
    rows.append(entry)
    write_csv(rows)
    return entry

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    check_daily_reset()
    if request.method == "POST":
        files = request.files.getlist("images")
        for f in files:
            if not f or not f.filename:
                continue
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(filepath)

            try:
                text, lines = ocr_extract(filepath)
                parsed = classify_lines(lines)
            except Exception as e:
                logging.exception("OCR error:")
                parsed = {"Name": "Not Found", "Company": "Not Found", "Phone No": "", "AWB No": ""}

            append_to_csv(parsed)

        return redirect(url_for("index"))

    table = read_csv()
    return render_template("index.html", table=table)

@app.route("/update_row/<sno>", methods=["POST"])
def update_row(sno):
    data = request.get_json() or {}
    filename = request.args.get("filename")
    
    if filename:
        file_path = os.path.join(app.config["ARCHIVE_FOLDER"], filename)
    else:
        file_path = DATA_FILE
        
    rows = read_csv(file_path)
    updated = False
    
    for row in rows:
        if row.get("S.No") == sno:
            # Update fields
            for k, v in data.items():
                if k in FIELDNAMES:
                    row[k] = v
            
            # Auto-fill if Roll No or Phone No changed (only for main data or if desired for archive too)
            # Assuming we want this behavior for archives too
            match = None
            if data.get("Roll No"):
                match = get_recipient_details(data["Roll No"])
            elif data.get("Phone No"):
                match = get_recipient_details(data["Phone No"])
            
            if match:
                row["Name"] = match["Name"] or row["Name"]
                row["Email"] = match["Email"] or row["Email"]
                row["Roll No"] = match["Roll No"] or row["Roll No"]
                row["Phone No"] = match["Phone No"] or row["Phone No"]
            
            updated = True
            break
            
    if updated:
        write_csv(rows, file_path)
        return jsonify({"message": f"Row {sno} updated."})
    return jsonify({"message": "Row not found"}), 404

@app.route("/update_status/<sno>/<status>", methods=["POST"])
def update_status(sno, status):
    filename = request.args.get("filename")
    
    if filename:
        file_path = os.path.join(app.config["ARCHIVE_FOLDER"], filename)
    else:
        file_path = DATA_FILE

    rows = read_csv(file_path)
    updated = False
    for row in rows:
        if row.get("S.No") == sno:
            row["Picked"] = status
            updated = True
            break
    if updated:
        write_csv(rows, file_path)
        return jsonify({"message": f"Row {sno} marked as {status}."})
    return jsonify({"message": "Row not found"}), 404

@app.route("/delete_row/<sno>", methods=["POST"])
def delete_row(sno):
    filename = request.args.get("filename")
    
    if filename:
        file_path = os.path.join(app.config["ARCHIVE_FOLDER"], filename)
    else:
        file_path = DATA_FILE

    rows = read_csv(file_path)
    rows = [r for r in rows if r.get("S.No") != sno]
    # Reindex
    for i, r in enumerate(rows, start=1):
        r["S.No"] = str(i)
    write_csv(rows, file_path)
    return jsonify({"message": f"Row {sno} deleted."})

@app.route("/add_row", methods=["POST"])
def add_row():
    rows = read_csv()
    s_no = str(len(rows) + 1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_compact = datetime.now().strftime("%Y%m%d")
    label_id = f"{today_compact}-{len(rows)+1:04d}"
    
    empty_row = {k: "" for k in FIELDNAMES}
    empty_row.update({
        "S.No": s_no,
        "Label ID": label_id,
        "Time": now,
        "Picked": "Not Picked",
        "Status": "Pending",
        "Mail Status": "Pending"
    })
    rows.append(empty_row)
    write_csv(rows)
    return redirect(url_for("index"))

@app.route("/send_bulk_pending", methods=["POST"])
def send_bulk_pending():
    rows = read_csv()
    pending_by_email = defaultdict(list)
    
    for row in rows:
        if row.get("Mail Status") != "Sent" and row.get("Email"):
            pending_by_email[row["Email"]].append(row)
            
    updated_snos = []
    failures = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for email, parcel_list in pending_by_email.items():
        # Construct Email
        subject = f"Parcel Arrival Notification - {len(parcel_list)} Package(s)"
        body = f"Hello,\n\nYou have {len(parcel_list)} parcel(s) waiting at the university gate.\n\n"
        for p in parcel_list:
            body += f"--- Parcel {p.get('Label ID')} ---\n"
            body += f"AWB: {p.get('AWB No')}\n"
            body += f"Company: {p.get('Company')}\n"
            body += f"Time: {p.get('Time')}\n\n"
        body += "Please collect them.\n\nRegards,\nSecurity"
        
        # Send Real Email
        success, error_msg = send_email_real(email, subject, body)
        
        if success:
            # Update Status
            for p in parcel_list:
                for r in rows:
                    if r["S.No"] == p["S.No"]:
                        r["Mail Status"] = "Sent"
                        r["Mail Time"] = now
                        updated_snos.append(r["S.No"])
        else:
            failures.append(f"{email}: {error_msg}")
                    
    write_csv(rows)
    
    msg = f"Sent notifications to {len(updated_snos)} parcels."
    if failures:
        msg += f"\n\nFailed ({len(failures)}):"
        for f in failures:
            msg += f"\n- {f}"
            
    return jsonify({
        "message": msg,
        "updated": updated_snos
    })

@app.route("/history")
def history():
    files = sorted(os.listdir(app.config["ARCHIVE_FOLDER"]), reverse=True)
    return render_template("history.html", files=files)

@app.route("/view_archive/<filename>")
def view_archive(filename):
    path = os.path.join(app.config["ARCHIVE_FOLDER"], filename)
    if not os.path.exists(path):
        return "File not found", 404
    rows = read_csv(path)
    return render_template("index.html", table=rows, archive_mode=True, filename=filename)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
