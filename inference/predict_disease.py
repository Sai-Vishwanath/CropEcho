import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# 1. Define the exact same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. List your 15 classes in the exact order they were trained
class_names = [
    'aphid', 'black_rust', 'blast', 'brown_rust', 'common_root_rot', 
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'mildew', 'mite', 
    'septoria', 'smut', 'stem_fly', 'tan_spot', 'yellow_rust'
]

def predict_wheat_disease(image_path, model_path='models/disease_resnet50.pth'):
    # Initialize a blank ResNet50 architecture
    model = models.resnet50()
    
    # Modify the final layer to match your 15 classes
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    
    # Load your trained weights into the architecture
    # weights_only=True is a security best practice for loading PyTorch models
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    
    model.eval() # Set to evaluation mode (This will work now!)
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    # Run the prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score

# --- Test the script locally ---
# --- Test the script locally ---
if __name__ == "__main__":
    import glob
    
    # Search for any file starting with 'test_disease' in the uploads folder
    possible_images = glob.glob("static/uploads/test_disease.*")
    
    if possible_images:
        # Grab the first matching file it finds (.png, .jpg, .jpeg, etc.)
        test_image = possible_images[0] 
        print(f"Found test image: {test_image}")
        
        disease, conf = predict_wheat_disease(test_image)
        print(f"Prediction: {disease} ({conf:.2f}% confidence)")
    else:
        print("Please place a test image named 'test_disease.jpg' or 'test_disease.png' in static/uploads/")