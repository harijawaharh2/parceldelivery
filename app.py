# app.py  -- use DEEPSEEK ONLY for OCR (robust wrapper)
import os
import re
import csv
import json
import logging
import shutil
import subprocess
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------- CONFIG (from environment) --------------------
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
ARCHIVE_FOLDER = os.environ.get("ARCHIVE_FOLDER", "archive")
DATA_FILE = os.environ.get("DATA_FILE", "data.csv")
CONTACT_FILE = os.environ.get("CONTACT_FILE", "contact.csv")
LAST_RUN_FILE = os.environ.get("LAST_RUN_FILE", "last_run_date.txt")

# Email config (set these in Render / env; do NOT commit secrets)
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))

# DeepSeek config (set one of these)
# Example: DEEPSEEK_CMD="deepseek-infer --model /models/deepseek"
DEEPSEEK_CMD = os.environ.get("DEEPSEEK_CMD", "").strip()
# OR point to a python script inside your repo: DEEPSEEK_SCRIPT="DeepSeek-OCR/infer.py"
DEEPSEEK_SCRIPT = os.environ.get("DEEPSEEK_SCRIPT", "").strip()
# As last option you can use Hugging Face Inference API (requires HF_TOKEN) and
# DEEPSEEK_HF_MODEL should be set to HF repo (e.g. "deepseek-ai/DeepSeek-OCR")
DEEPSEEK_HF_MODEL = os.environ.get("DEEPSEEK_HF_MODEL", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Timeout for calling external DeepSeek processes (seconds)
DEEPSEEK_TIMEOUT = int(os.environ.get("DEEPSEEK_TIMEOUT", 60))

# CSV fields
FIELDNAMES = [
    "S.No", "Label ID", "Roll No", "Name", "Company", "AWB No",
    "Email", "Phone No", "Time", "Parcel No", "Picked", "Signature",
    "Status", "Mail Status", "Mail Time"
]

# Create folders
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(ARCHIVE_FOLDER).mkdir(parents=True, exist_ok=True)

# Flask app
app = Flask(__name__, static_folder=None)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ARCHIVE_FOLDER"] = ARCHIVE_FOLDER

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek-app")

# -------------------- EMAIL UTIL --------------------
def send_email_real(to_email, subject, body):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        logger.warning("SMTP not configured; skipping email send.")
        return False, "SMTP not configured"
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.starttls()
        server.login(SMTP_EMAIL.strip(), SMTP_PASSWORD.strip())
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        server.quit()
        logger.info("Email sent to %s", to_email)
        return True, None
    except Exception as e:
        logger.exception("Failed to send email")
        return False, str(e)

# -------------------- DeepSeek OCR wrapper (DEEPSEEK ONLY) --------------------
def deepseek_call_cli(image_path):
    """
    Execute DEEPSEEK_CMD with image path appended (if DEEPSEEK_CMD provided).
    e.g. DEEPSEEK_CMD="deepseek-infer --model /models/deepseek"
    Actual command executed: <DEEPSEEK_CMD> --image <image_path>
    """
    if not DEEPSEEK_CMD:
        return None
    # build list - split DEEPSEEK_CMD to argv list safely
    try:
        base_args = DEEPSEEK_CMD.split()
        # Common CLI flag names vary; we'll append common flags and try multiple shapes
        attempts = []
        # 1) append explicit flag
        attempts.append(base_args + ["--image", image_path])
        # 2) append positional
        attempts.append(base_args + [image_path])
        # 3) explicit -i
        attempts.append(base_args + ["-i", image_path])
        for args in attempts:
            try:
                logger.debug("Trying DeepSeek CLI: %s", " ".join(args))
                proc = subprocess.run(args, capture_output=True, text=True, timeout=DEEPSEEK_TIMEOUT)
                if proc.returncode != 0:
                    logger.debug("DeepSeek CLI returned %d; stderr: %.200s", proc.returncode, proc.stderr)
                    continue
                stdout = proc.stdout.strip()
                if stdout:
                    # Try parse JSON first (some deepseek scripts output JSON)
                    try:
                        parsed = json.loads(stdout)
                        # Common key names -> try to extract text
                        for key in ("text", "ocr_text", "result", "pred", "output"):
                            if isinstance(parsed, dict) and key in parsed:
                                val = parsed[key]
                                return str(val)
                        # if parsed is stringish
                        return stdout
                    except Exception:
                        # not JSON, return raw stdout
                        return stdout
            except subprocess.TimeoutExpired:
                logger.warning("DeepSeek CLI attempt timed out for args: %s", args)
                continue
            except FileNotFoundError:
                logger.error("DeepSeek CLI command not found: %s", args[0])
                break
        return None
    except Exception:
        logger.exception("Error while attempting DeepSeek CLI")
        return None

def deepseek_call_script(image_path):
    """
    Try running a local python script (DEEPSEEK_SCRIPT). We'll try:
    python <script> --image <image_path>
    """
    if not DEEPSEEK_SCRIPT:
        return None
    try:
        script_args = ["python", DEEPSEEK_SCRIPT, "--image", image_path]
        logger.debug("Trying DeepSeek script: %s", " ".join(script_args))
        proc = subprocess.run(script_args, capture_output=True, text=True, timeout=DEEPSEEK_TIMEOUT)
        if proc.returncode != 0:
            logger.debug("DeepSeek script returned %d; stderr: %.200s", proc.returncode, proc.stderr)
            # try positional
            proc2 = subprocess.run(["python", DEEPSEEK_SCRIPT, image_path], capture_output=True, text=True, timeout=DEEPSEEK_TIMEOUT)
            if proc2.returncode != 0:
                logger.debug("DeepSeek script (positional) returned %d", proc2.returncode)
                return None
            out = proc2.stdout.strip()
        else:
            out = proc.stdout.strip()
        if not out:
            return None
        try:
            parsed = json.loads(out)
            for key in ("text", "ocr_text", "result", "pred"):
                if key in parsed:
                    return str(parsed[key])
            return out
        except Exception:
            return out
    except FileNotFoundError:
        logger.error("Python interpreter not found when invoking DeepSeek script.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("DeepSeek script timed out.")
        return None
    except Exception:
        logger.exception("Error calling DeepSeek script")
        return None

def deepseek_call_hf_inference(image_path):
    """
    Last-resort: call Hugging Face Inference API for a model (DEEPSEEK_HF_MODEL).
    Requires HF_TOKEN in env. Will POST the image bytes to the inference endpoint.
    """
    if not DEEPSEEK_HF_MODEL or not HF_TOKEN:
        return None
    try:
        import requests
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        url = f"https://api-inference.huggingface.co/models/{DEEPSEEK_HF_MODEL}"
        with open(image_path, "rb") as fh:
            data = fh.read()
        logger.debug("Calling HF inference for model %s", DEEPSEEK_HF_MODEL)
        resp = requests.post(url, headers=headers, data=data, timeout=DEEPSEEK_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("HF inference returned %d: %.200s", resp.status_code, resp.text)
            return None
        # HF may return JSON with outputs or plain text
        try:
            j = resp.json()
            # try to extract fields
            if isinstance(j, dict):
                for k in ("text", "ocr_text", "result"):
                    if k in j:
                        return str(j[k])
            # fallback to raw text if available
            if isinstance(j, str):
                return j
            # or join textual items
            if isinstance(j, list):
                texts = []
                for part in j:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                if texts:
                    return "\n".join(texts)
            return json.dumps(j)
        except Exception:
            return resp.text
    except Exception:
        logger.exception("HF inference call failed")
        return None

def deepseek_ocr(image_path):
    """
    Master wrapper: calls in order:
     1) DEEPSEEK_CMD CLI (if set)
     2) DEEPSEEK_SCRIPT python script (if set)
     3) Hugging Face Inference (if set)
    Returns (raw_text, lines_list). If no method succeeds, returns ("", []).
    """
    logger.info("Running DeepSeek OCR for %s", image_path)
    # 1) CLI
    if DEEPSEEK_CMD:
        try:
            out = deepseek_call_cli(image_path)
            if out:
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                return out, lines
        except Exception:
            logger.exception("DeepSeek CLI attempt failed unexpectedly.")
    # 2) script
    if DEEPSEEK_SCRIPT:
        try:
            out = deepseek_call_script(image_path)
            if out:
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                return out, lines
        except Exception:
            logger.exception("DeepSeek script attempt failed unexpectedly.")
    # 3) HF inference
    if DEEPSEEK_HF_MODEL and HF_TOKEN:
        try:
            out = deepseek_call_hf_inference(image_path)
            if out:
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                return out, lines
        except Exception:
            logger.exception("DeepSeek HF attempt failed unexpectedly.")
    # If nothing worked, log and return blank (do NOT fallback to other OCRs)
    logger.error("DeepSeek not available or all attempts failed. Check DEEPSEEK_CMD/DEEPSEEK_SCRIPT/DEEPSEEK_HF_MODEL.")
    return "", []

# -------------------- CLASSIFICATION/HEURISTICS --------------------
def classify_lines(lines):
    name = company = phone = awb = rollno = None
    cleaned = [re.sub(r'[^a-zA-Z0-9\s,+-.]', '', l).strip() for l in lines if len(l.strip()) > 2]
    for line in cleaned:
        # AWB (10-15 digits)
        if not awb and re.search(r'\b\d{10,15}\b', line):
            awb = re.search(r'\b\d{10,15}\b', line).group(0)
            continue
        # PHONE (India-style check, generic 10-digit)
        if not phone and re.search(r'(\+?91|0)?\s?\d{10}\b', line):
            phone = re.search(r'\d{10}\b', line).group(0)
            continue
        # ROLL NO heuristic
        if not rollno and re.search(r'\b\d{2}[A-Z0-9]{8}\b', line, re.IGNORECASE):
            rollno = re.search(r'\b\d{2}[A-Z0-9]{8}\b', line, re.IGNORECASE).group(0)
            continue
        # COMPANY keywords
        if not company and any(x in line.lower() for x in ["flipkart", "ekart", "delhivery", "amazon", "bluedart", "xpressbees", "ecom", "shadowfax"]):
            company = line
            continue
        # NAME (simple)
        if not name and re.match(r'^[A-Za-z][A-Za-z\s.]{2,40}$', line):
            name = line.strip()
    return {
        "Name": name or "",
        "Company": company or "",
        "Phone No": phone or "",
        "AWB No": awb or "",
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
                archive_path = os.path.join(ARCHIVE_FOLDER, archive_name)
                shutil.copy(DATA_FILE, archive_path)
                logger.info("Archived data to %s", archive_path)
                with open(DATA_FILE, "w", newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
    with open(LAST_RUN_FILE, "w") as f:
        f.write(today_str)

def append_to_csv(data):
    check_daily_reset()
    rows = read_csv()
    today_compact = datetime.now().strftime("%Y%m%d")
    serial = len(rows) + 1
    label_id = f"{today_compact}-{serial:04d}"
    s_no = str(len(rows) + 1)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                raw_text, lines = deepseek_ocr(filepath)
                parsed = classify_lines(lines)
            except Exception as e:
                logger.exception("DeepSeek OCR error for %s", filepath)
                parsed = {"Name": "", "Company": "", "Phone No": "", "AWB No": "", "Roll No": ""}
            append_to_csv(parsed)
        return redirect(url_for("index"))
    table = read_csv()
    return render_template("index.html", table=table)

@app.route("/update_row/<sno>", methods=["POST"])
def update_row(sno):
    data = request.get_json() or {}
    filename = request.args.get("filename")
    file_path = os.path.join(app.config["ARCHIVE_FOLDER"], filename) if filename else DATA_FILE
    rows = read_csv(file_path)
    updated = False
    for row in rows:
        if row.get("S.No") == sno:
            for k, v in data.items():
                if k in FIELDNAMES:
                    row[k] = v
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
    file_path = os.path.join(app.config["ARCHIVE_FOLDER"], filename) if filename else DATA_FILE
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
    file_path = os.path.join(app.config["ARCHIVE_FOLDER"], filename) if filename else DATA_FILE
    rows = read_csv(file_path)
    rows = [r for r in rows if r.get("S.No") != sno]
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
        subject = f"Parcel Arrival Notification - {len(parcel_list)} Package(s)"
        body = f"Hello,\n\nYou have {len(parcel_list)} parcel(s) waiting at the university gate.\n\n"
        for p in parcel_list:
            body += f"--- Parcel {p.get('Label ID')} ---\n"
            body += f"AWB: {p.get('AWB No')}\n"
            body += f"Company: {p.get('Company')}\n"
            body += f"Time: {p.get('Time')}\n\n"
        body += "Please collect them.\n\nRegards,\nSecurity"
        success, error_msg = send_email_real(email, subject, body)
        if success:
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
    return jsonify({"message": msg, "updated": updated_snos})

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

# -------------------- RUN --------------------
if __name__ == "__main__":
    # In production use gunicorn; this is for local/dev testing.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG","0") == "1")
