import numpy as np
import torch
import clip


def clip_image_similarity(real_image, generated_image, clip_model_name="ViT-B/32", device="cpu"):

    # Load the CLIP model
    model, preprocess = clip.load(name=clip_model_name, device=device)

    # Preprocess the images using the CLIP-preferred transformations
    real_image = preprocess(real_image).unsqueeze(0).to(device)
    generated_image = preprocess(generated_image).unsqueeze(0).to(device)

    # Encode the images using the CLIP model
    with torch.no_grad():
        real_features = model.encode_image(real_image)
        generated_features = model.encode_image(generated_image)

        # normalize images
        real_features = real_features / real_features.norm(dim=1, keepdim=True)

    # Compute the cosine similarity between the real and generated image features
    similarity = torch.nn.functional.cosine_similarity(real_features, generated_features).item()

    return similarity


    

class CLIPMetrics:

    def __init__(self, clip_model_name="ViT-B/32", device="cpu"):
        self.device = device
        # Load the CLIP model
        self.clip, self.preprocess = clip.load(name=clip_model_name, device=device)

    def predict_shape(self, image, labels):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        labels = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.clip(image, labels)
        
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs
        


def binary_iou(image1, image2, treshold=0.5):
    """
    Calculates the Intersection over Union (IoU) of two binary images.


    Parameters:
    image1 (np.array): The first input image. Must be the same shape as image2.
    image2 (np.array): The second input image. Must be the same shape as image1.
    treshold (float, optional): The threshold for converting input images to binary images. Defaults to 0.5.

    Returns:
    float: The IoU of the two binary images.

    """
    # Ensure the images are the same shape
    assert image1.shape == image2.shape, "Images must have the same shape."

    # Convert images to boolean arrays
    image1 = image1 >= treshold
    image2 = image2 >= treshold
    image1_bool = image1.astype(bool)
    image2_bool = image2.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(image1_bool, image2_bool)
    union = np.logical_or(image1_bool, image2_bool)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou
