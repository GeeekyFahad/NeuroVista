import torch
from PIL import Image
from torchvision import transforms
from model import AlzheimerResNet
import sys

classes = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return

    img_path = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AlzheimerResNet(num_classes=4)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    image = load_image(img_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    print("Prediction:", classes[pred])
    print("Confidence:", round(confidence, 3))

if __name__ == "__main__":
    main()
