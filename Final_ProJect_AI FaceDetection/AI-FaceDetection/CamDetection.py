import torch
import numpy as np
import cv2
import os
import sqlite3
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from datetime import datetime
import random
import string

# 🛠 พาธ
base_dir = "D:/FaceDetection"
embedding_dir = os.path.join(base_dir, "embeddings")
os.makedirs(base_dir, exist_ok=True)
os.makedirs(embedding_dir, exist_ok=True)

# 🔌 โหลดโมเดล
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained=None).eval().to(device)
checkpoint = torch.load("facenet_checkpoint34.pth", map_location=device)
model.load_state_dict({k: v for k, v in checkpoint['model_state'].items() if 'logits' not in k})
print("✅ โหลดโมเดลเรียบร้อย")

# ✅ สร้างฐานข้อมูล
def init_databases():
    with sqlite3.connect(os.path.join(base_dir, "employees.db")) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS employees (
            emp_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, image_path TEXT)''')

    with sqlite3.connect(os.path.join(base_dir, "attendance.db")) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
            emp_id TEXT, first_name TEXT, last_name TEXT, image_path TEXT,
            datetime TEXT, status TEXT)''')

# 🎲 สุ่มรหัสพนักงาน
def generate_emp_id():
    return ''.join(random.choices(string.digits, k=10))

# 🎯 ดึง embedding
def get_embedding(pil_image):
    face = mtcnn(pil_image)
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(face).cpu().numpy().squeeze(0)

# 💾 บันทึกพนักงานใหม่ พร้อมเช็ค emp_id ซ้ำ
def save_new_employee(frame_rgb):
    # สุ่มรหัสพนักงานไม่ให้ซ้ำ
    while True:
        emp_id = generate_emp_id()
        with sqlite3.connect(os.path.join(base_dir, "employees.db")) as conn:
            result = conn.execute("SELECT emp_id FROM employees WHERE emp_id = ?", (emp_id,)).fetchone()
            if result is None:
                break  # หยุดเมื่อไม่มี ID ซ้ำ

    first_name = input("ชื่อ: ")
    last_name = input("นามสกุล: ")
    pil_image = Image.fromarray(frame_rgb)
    embedding = get_embedding(pil_image)

    if embedding is None:
        print("❌ ไม่พบใบหน้า")
        return

    # สร้างโฟลเดอร์ของพนักงาน
    folder = os.path.join(base_dir, emp_id)
    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, f"{emp_id}.jpg")

    # บันทึกภาพและ embedding
    Image.fromarray(frame_rgb).save(image_path)
    np.save(os.path.join(embedding_dir, f"{emp_id}_embedding.npy"), embedding)

    with sqlite3.connect(os.path.join(base_dir, "employees.db")) as conn:
        conn.execute("INSERT INTO employees VALUES (?, ?, ?, ?)",
                     (emp_id, first_name, last_name, image_path))
    print(f"✅ บันทึกพนักงาน {emp_id} แล้ว")


# 📌 โหลด embedding ทั้งหมด
def load_known_faces():
    faces = {}
    with sqlite3.connect(os.path.join(base_dir, "employees.db")) as conn:
        rows = conn.execute("SELECT * FROM employees").fetchall()
        for emp_id, fname, lname, img_path in rows:
            emb_path = os.path.join(embedding_dir, f"{emp_id}_embedding.npy")
            if os.path.exists(emb_path):
                faces[emp_id] = {
                    "embedding": np.load(emb_path),
                    "first_name": fname,
                    "last_name": lname,
                    "image_path": img_path
                }
    return faces

# ✏️ บันทึกเวลาเข้าออก
def log_attendance(emp_id, fname, lname, img_path, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(os.path.join(base_dir, "attendance.db")) as conn:
        conn.execute("INSERT INTO attendance VALUES (?, ?, ?, ?, ?, ?)",
                     (emp_id, fname, lname, img_path, now, status))
    print(f"📅 {status.upper()} - {fname} {lname} [{emp_id}] @ {now}")

# 🟢 เริ่มระบบกล้อง DroidCam
def run_face_recognition():
    known_faces = load_known_faces()
    droidcam_url = "http://192.168.1.103:4747/video"
    cap = cv2.VideoCapture(droidcam_url)
    print("📷 เริ่มจับภาพ (กด S=บันทึก, I=เข้า, O=ออก, Q=ออกโปรแกรม)")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        emb = get_embedding(pil_image)

        text_display = "No Match"
        matched_id = matched_name = ""
        if emb is not None:
            min_dist, threshold = float("inf"), 1.2
            for emp_id, data in known_faces.items():
                dist = np.linalg.norm(emb - data['embedding'])
                if dist < min_dist:
                    min_dist = dist
                    if dist < threshold:
                        matched_id = emp_id
                        matched_name = f"{data['first_name']} {data['last_name']}"
                        text_display = f"{matched_id} | {matched_name}"

        # แสดงผลบนจอ
        now = datetime.now().strftime("%H:%M:%S")
        display_text = f"{text_display} | {now}"
        color = (0, 255, 0) if matched_id else (0, 0, 255)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("📍 Face Attendance", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_new_employee(rgb)
            known_faces = load_known_faces()  # reload
        elif key == ord('i') and matched_id:
            data = known_faces[matched_id]
            log_attendance(matched_id, data['first_name'], data['last_name'], data['image_path'], "in")
        elif key == ord('o') and matched_id:
            data = known_faces[matched_id]
            log_attendance(matched_id, data['first_name'], data['last_name'], data['image_path'], "out")

    cap.release()
    cv2.destroyAllWindows()

# 🚀 เรียกใช้งาน
init_databases()
run_face_recognition()
