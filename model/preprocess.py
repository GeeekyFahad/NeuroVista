import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import random

# CLAHE (Contrast Enhancement)
def apply_clahe(pil_img):
    gray = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced).convert("RGB")

# Auto Crop Foreground
def auto_crop(pil_img, threshold=10):
    gray = np.array(pil_img.convert("L"))
    coords = np.where(gray > threshold)

    if coords[0].size == 0:
        return pil_img

    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()

    return pil_img.crop((x1, y1, x2, y2))

# 2D Skull Stripping (Approx)
def skull_strip_2d(pil_img):
    img = np.array(pil_img.convert("L"))

    # Blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return pil_img

    # Largest contour = brain region
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

    stripped = cv2.bitwise_and(img, img, mask=mask)

    return Image.fromarray(stripped).convert("RGB")

# Final Transform Pipeline
def get_transforms(augment=False):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)

# Main Preprocess Function
def preprocess_image(img_path, augment=False):
    img = Image.open(img_path).convert("RGB")

    # Step 1: Auto-crop
    img = auto_crop(img)

    # Step 2: Skull stripping (approx)
    img = skull_strip_2d(img)

    # Step 3: CLAHE enhancement
    img = apply_clahe(img)

    # Step 4: Torch transforms
    transform = get_transforms(augment)
    img = transform(img)

    return img
