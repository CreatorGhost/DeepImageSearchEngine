import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from vector_data import RedisImageSearch
from embed import RedisVectorDatabase
import numpy as np
# Initialize the pre-trained ResNet model and transformation pipeline
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'embedding'})

def generate_image_embedding(image_path):
    """
    Generate a vector embedding for an image using a pre-trained ResNet model.

    """
    image = Image.open(image_path)  # Load the image
    image = transform(image)  # Preprocess the image
    image = image.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  # No need to compute gradients
        features = feature_extractor(image)  # Extract features
        embedding = features['embedding'].squeeze()  # Remove extra dimensions

    embedding = torch.flatten(embedding).numpy()
    return embedding


if __name__ == "__main__":
    # Initialize your Redis vector database connection
    db = RedisVectorDatabase(host='localhost', port=6379, password=None)
    index_name = "WemageEmbeddings"
    vector_dimension = 512  # Dimension of your embeddings
    
    # Create an index for storing vectors if it doesn't already exist
    # This step is required once, comment it out after the index is created
    db.create_index(index_name, vector_dimension)
    
    image_paths = ["images/image1.jpg", "images/image4.jpg", "images/image3.jpg", "images/diwakar1.jpg", "images/diwakar2.jpg"]
    for image_path in image_paths:
        embedding = generate_image_embedding(image_path)
        file_name = image_path.split("/")[-1]
        
        # Convert embedding to the expected format (numpy array of type np.float32)
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Prepare metadata (adjust according to your needs)
        metadata = {"name": file_name, "type": "image/jpeg"}
        
        # Store the embedding and metadata
        db.store_vector(index_name, file_name, embedding, metadata)

    # Generate embedding for a query image
    query_image_path = "images/image2.jpg"
    query_embedding = generate_image_embedding(query_image_path)
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Perform a similarity search
    similar_images = db.search_similar_vectors(index_name, query_embedding,threshold = 0.2, top_k=3)
    print(similar_images)