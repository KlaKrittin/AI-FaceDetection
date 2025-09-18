import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm

# โหลดโมเดล FaceNet (VGGFace2)
model = InceptionResnetV1(pretrained='vggface2', classify=False).train()

# Freeze ชั้นบางส่วน
for name, param in model.named_parameters():
    if 'conv' in name:
        param.requires_grad = False

# ใช้ GPU ถ้ามี
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")
print(torch.cuda.is_available())  # ตรวจสอบว่า CUDA ใช้งานได้หรือไม่
print(torch.cuda.current_device())  # ดูว่า GPU ตัวไหนที่ใช้งานอยู่
print(torch.cuda.get_device_name(0))  # ตรวจสอบชื่อ GPU ของคุณ

# Loss Function
loss_fn = nn.TripletMarginLoss(margin=1.0)

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)

# เพิ่ม Data Augmentation รวม Gaussian Blur และ Brightness adjustment
transform = transforms.Compose([
    transforms.RandomResizedCrop(160, scale=(0.9, 1.0)),  # ลดขนาด crop น้อยลง
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # เพิ่ม Rotation ±15°
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),  # ลด brightness เป็น 0.1
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # ลด blur ให้อ่อนลง
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # Gaussian Noise (ลดลงจาก 0.05 → 0.03)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])


dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

# โหลดโมเดลถ้ามี checkpoint
checkpoint_path = "facenet_checkpoint.pth"
start_epoch = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"🔄 โหลดโมเดลจาก '{checkpoint_path}', เริ่มที่ Epoch {start_epoch}")
else:
    print("🚀 เริ่มฝึกใหม่ตั้งแต่ต้น")

# ฟังก์ชันเลือก Triplets
def get_triplets(images, labels):
    anchors, positives, negatives = [], [], []
    unique_labels = list(set(labels.tolist()))

    for label in unique_labels:
        same_class_indices = torch.where(labels == label)[0]
        different_class_indices = torch.where(labels != label)[0]

        if len(same_class_indices) > 1 and len(different_class_indices) > 0:
            try:
                anchor, positive = random.sample(list(same_class_indices), 2)
                negative = random.choice(list(different_class_indices))
                anchors.append(images[anchor])
                positives.append(images[positive])
                negatives.append(images[negative])
            except ValueError:
                continue  # ข้ามถ้าเลือกไม่ได้

    if len(anchors) == 0:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


# ฟังก์ชันการฝึก
def train_model():
    epochs = 50
    global start_epoch  
    best_loss = float('inf')  # ค่าการ loss ที่ดีที่สุด
    patience = 5 # จำนวน epoch ที่จะรอถ้าการ loss ไม่ลดลง
    trigger_times = 0  # ตัวนับจำนวน epoch ที่ไม่ลดลง

    for epoch in range(start_epoch, epochs):
        print(f"🟡 เริ่ม Epoch {epoch + 1}/{epochs}")

        model.train()
        total_loss = 0

        # ใช้ tqdm สำหรับแสดง progress bar
        with tqdm(dataloader, desc=f"Epoch {epoch+1}", total=len(dataloader)) as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                if len(images) < 2:
                    continue

                optimizer.zero_grad()
                anchors, positives, negatives = get_triplets(images, labels)

                if anchors is None or len(anchors) == 0:
                    continue

                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

                if len(anchors) == 1:
                    model.eval()
                else:
                    model.train()

                # คำนวณ Embeddings
                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)

                model.train()

                # คำนวณ Loss
                loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # อัปเดต progress bar
                pbar.set_postfix(loss=loss.item())

        print(f"🟢 Epoch {epoch + 1}/{epochs} เสร็จสิ้น - Total Loss: {total_loss:.4f}\n")

        # เช็คว่า loss ลดลงหรือไม่
        if total_loss < best_loss:
            best_loss = total_loss
            trigger_times = 0  # รีเซ็ตเมื่อ loss ลดลง
            # ✅ บันทึกโมเดลทุกครั้งที่ loss ลดลง
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 บันทึกโมเดลที่ '{checkpoint_path}'")
        else:
            trigger_times += 1
            print(f"🔴 ไม่พบการลดลงของ loss เป็นเวลา {trigger_times} epoch")

        # ถ้าการ loss ไม่ลดลงภายใน patience epochs ก็จะหยุดการฝึก
        if trigger_times >= patience:
            print("🚨 Early stopping เนื่องจากการ loss ไม่ลดลงในหลายๆ epoch")
            break

    print("✅ ฝึกเสร็จสิ้น!")

train_model()