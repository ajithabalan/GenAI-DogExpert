
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the trained model & processor
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=31, 
    ignore_mismatched_sizes=True 
)
model.load_state_dict(torch.load("dog_breed_model.pth", map_location=device), strict=False)  
model.to(device)
model.eval()


image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

#  Define breed names
BREED_NAMES = {
    0: "Chihuahua",
    1: "Shih-Tzu",
    2: "Toy Terrier",
    3: "Italian Greyhound",
    4: "American Staffordshire Terrier",
    5: "Bedlington Terrier",
    6: "Border Terrier",
    7: "Standard Schnauzer",
    8: "Tibetan Terrier",
    9: "Silky Terrier",
    10: "Flat-Coated Retriever",
    11: "Curly-Coated Retriever",
    12: "Golden Retriever",
    13: "Labrador Retriever",
    14: "Chesapeake Bay Retriever",
    15: "Collie",
    16: "Rottweiler",
    17: "German Shepherd",
    18: "Doberman",
    19: "Boxer",
    20: "Bull Mastiff",
    21: "Tibetan Mastiff",
    22: "French Bulldog",
    23: "Great Dane",
    24: "Saint Bernard",
    25: "Siberian Husky",
    26: "Pug",
    27: "Pomeranian",
    28: "Toy Poodle",
    29: "Miniature Poodle",
    30: "Standard Poodle"
}


def predict_breed(image_path):
    """Predict the breed of the dog from an image."""
    image = Image.open(image_path).convert("RGB")  
    inputs = image_processor(images=image, return_tensors="pt").to(device)  

    with torch.no_grad():
        outputs = model(**inputs).logits
    predicted_class = torch.argmax(outputs, dim=1).item()
    
    return BREED_NAMES.get(predicted_class, "Unknown Breed")
