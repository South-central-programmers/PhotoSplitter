import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

torch.cuda.set_device(0)

model = YOLO(
    "training_results/weights/face_detection_without_cutout/face_detection_without_cutout_best.pt"
)


def process_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in tqdm(files, desc=f"Checking folder {root}"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)

                results = model.predict(image, verbose=False)

                if (
                    not results[0].boxes.xyxy.tolist()
                    or len(results[0].boxes.xyxy.tolist()) != 1
                ):
                    continue

                face_data = results[0].boxes.xyxy.tolist()
                if len(face_data[0]) != 4:
                    continue

                x1, y1, x2, y2 = face_data[0]
                face = image.crop((x1, y1, x2, y2))
                image_name, image_ext = os.path.splitext(file)
                new_image_name = f"{image_name}_e{image_ext}"
                new_image_path = os.path.join(root, new_image_name)

                os.remove(image_path)

                face.save(new_image_path)


folders = ["data/face_similarity/train", "data/face_similarity/val"]

for folder in folders:
    process_images(folder)
