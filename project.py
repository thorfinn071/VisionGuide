import torch
from ultralytics import YOLO
import cv2

# Загружаем TorchScript модель
model = torch.jit.load("yolov8n.torchscript")
model.eval()

# Камера
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO принимает RGB изображение
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Запуск детекции
    results = model(img_rgb)

    # results — обычная структура Ultralytics, можно обработать как раньше
    # ... твоя логика для TTS и отрисовки
