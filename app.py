import os
import time
import json
import base64
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import extract

# Flask App Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parcel_notify.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24)

db = SQLAlchemy(app)

# --- MODELS ---
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    parcels = db.relationship('Parcel', backref='student', lazy=True)

class Parcel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    serial_no = db.Column(db.String(20), unique=True, nullable=False)
    awb = db.Column(db.String(100), nullable=True)
    label_name = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=True)
    status = db.Column(db.String(20), default='Arrived') # Arrived, Picked Up
    picked = db.Column(db.Boolean, default=False)
    mail_sent = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- GEMINI API INTEGRATION ---
API_KEY = "" # The execution environment provides the key

def call_gemini_ocr(base64_image):
    """Calls Gemini API with exponential backoff as per environment rules."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
    
    prompt = "Extract recipient name, phone number, tracking number (AWB), and student roll number from this parcel label. Return as JSON."
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": base64_image}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "phone": {"type": "STRING"},
                    "awb": {"type": "STRING"},
                    "rollNo": {"type": "STRING"}
                }
            }
        }
    }

    retries = 0
    delays = [1, 2, 4, 8, 16]
    
    while retries < 5:
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text_content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                return json.loads(text_content)
            retries += 1
            if retries < 5:
                time.sleep(delays[retries])
        except Exception:
            retries += 1
            if retries < 5:
                time.sleep(delays[retries])
    
    return None

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def scan_label():
    data = request.json
    base64_image = data.get('image') # Expects image without prefix
    
    if not base64_image:
        return jsonify({"error": "No image provided"}), 400
        
    ocr_result = call_gemini_ocr(base64_image)
    if not ocr_result:
        return jsonify({"error": "AI processing failed"}), 500
        
    # Attempt to match student
    match = None
    if ocr_result.get('rollNo'):
        match = Student.query.filter_by(roll_no=ocr_result['rollNo']).first()
    
    if not match and ocr_result.get('name'):
        match = Student.query.filter(Student.name.ilike(f"%{ocr_result['name']}%")).first()
        
    response = {
        "ocr": ocr_result,
        "matched_student": {
            "id": match.id,
            "name": match.name,
            "email": match.email
        } if match else None
    }
    return jsonify(response)

@app.route('/api/parcels', methods=['GET', 'POST'])
def handle_parcels():
    if request.method == 'POST':
        data = request.json
        # Generate Serial Number
        count = Parcel.query.count()
        serial_no = f"UNI-{datetime.now().year}-{str(count + 1).zfill(4)}"
        
        new_parcel = Parcel(
            serial_no=serial_no,
            awb=data.get('awb', 'N/A'),
            label_name=data.get('name', 'Unknown'),
            student_id=data.get('student_id'),
            status='Arrived'
        )
        db.session.add(new_parcel)
        db.session.commit()
        return jsonify({"message": "Parcel logged", "serial_no": serial_no})
    
    # GET: Fetch grouped history
    parcels = Parcel.query.order_by(Parcel.timestamp.desc()).all()
    grouped = {}
    for p in parcels:
        month_year = p.timestamp.strftime("%B %Y")
        if month_year not in grouped:
            grouped[month_year] = []
        grouped[month_year].append({
            "id": p.id,
            "serial_no": p.serial_no,
            "name": p.student.name if p.student else p.label_name,
            "status": p.status,
            "picked": p.picked,
            "awb": p.awb,
            "date": p.timestamp.strftime("%Y-%m-%d")
        })
    return jsonify(grouped)

@app.route('/api/students', methods=['GET', 'POST'])
def handle_students():
    if request.method == 'POST':
        data = request.json
        new_student = Student(
            name=data['name'],
            roll_no=data['roll_no'],
            email=data['email'],
            phone=data['phone']
        )
        db.session.add(new_student)
        db.session.commit()
        return jsonify({"message": "Student registered"})
    
    students = Student.query.all()
    return jsonify([{
        "id": s.id,
        "name": s.name,
        "roll_no": s.roll_no,
        "email": s.email,
        "phone": s.phone
    } for s in students])

@app.route('/api/parcels/<int:pid>/pick', methods=['POST'])
def pick_parcel(pid):
    parcel = Parcel.query.get_or_404(pid)
    parcel.picked = not parcel.picked
    parcel.status = "Picked Up" if parcel.picked else "Arrived"
    db.session.commit()
    return jsonify({"status": parcel.status})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
