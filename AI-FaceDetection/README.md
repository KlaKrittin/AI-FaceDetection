ต้องโหลด python เวอร์ชัน 3.11.0 ในเว็ป
https://www.python.org/downloads/release/python-3110/และทำการติดตั้ง

ทำการแตกไฟล์Final_ProJect_AI FaceDetection และทำการโหลดดาต้าเซตจาก https://www.kaggle.com/datasets/hearfool/vggface2 ทำการแตกไฟล์ และนำดาต้าเซตมาใส่ ในโฟล์เดอร์ Final_ProJect_AI 
FaceDetection และทำการเปลี่ยนชื่อดาต้าเซตเป็นdataset เปิดVs code 

กด Ctrl+Shift+P → พิมพ์ "Python: Select Interpreter"
ทำการเลือก 3.11.0

ทำการเลือก Terminal และกด New Terminal หรือกด Ctrl + Shift + ~

ทำการสร้าง Virtual Environment โดยพิมท์ python -m venv venv ใน Terminal
ทำการ Activate Virtual Environment โดยพิมท์ .\venv\Scripts\activate ใน Terminal

ติดตั้งแพ็กเกจทั้งหมดจาก requirements.txt
pip install -r requirements.txt

ถ้าคุณเจอปัญหา Execution Policy Error ตอน activate venv ให้รัน
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

และทำการ python -m venv venv หากVenvยังไม่ถูกสร้าง และ .\venv\Scripts\activate อีกครั้ง 

ทำการติดตั้งแพ็กเกจทั้งหมดจาก requirements.txt โดยการ pip install -r requirements.txt

จากนั้นสามารถรันโปรแกรมได้ตามปกติ