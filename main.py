from dataset import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

img_path = input("Enter Image File Path: ")


image = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0).to(device)
model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
    probabilities = F.softmax(prediction, dim=1)
    threshold = 0.1
    above_threshold_indices = torch.where(probabilities > threshold)[1]
    above_threshold_probs = probabilities[0][above_threshold_indices]
Class_names = [
    "Crimp", "Sloper", "Jug", "Pinch", "Pocket", "Foot"
]
plt.imshow(image)
plt.axis('off')
plt.show()
print("Predictions above threshold:")
for idx, prob in zip(above_threshold_indices, above_threshold_probs):
    class_name = Class_names[idx.item()]
    print(f"{class_name}: {prob.item()*100:.4f}%")