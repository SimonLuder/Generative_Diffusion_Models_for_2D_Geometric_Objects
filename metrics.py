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
    
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    Calculate Intersection over Union (IoU) for RGB images.

    Parameters:
    outputs (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of images all of the same size.
    labels (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of images all of the same size.
    threshold (float): threshold for converting grayscale images to binary masks.

    Returns:
    Tensor: IoU for each image in the batch.
    """
    # # Convert RGB images to grayscale
    # outputs_gray = outputs.mean(dim=1)
    # labels_gray = labels.mean(dim=1)

    # Convert grayscale images to binary masks
    outputs_bin = (outputs > threshold)
    labels_bin = (labels > threshold)

    intersection = (outputs_bin & labels_bin).sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs_bin | labels_bin).sum((1, 2, 3))         # Will be zero if both are 0

    iou = intersection / union  # We smooth our division to avoid 0/0

    return iou

def center_distance_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    Calculate Intersection over Union (IoU) for RGB images.

    Parameters:
    outputs (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of images all of the same size.
    labels (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of images all of the same size.
    threshold (float): threshold for converting grayscale images to binary masks.

    Returns:
    Tensor: IoU for each image in the batch.
    """
    # # Convert RGB images to grayscale
    # outputs_gray = outputs.mean(dim=1)
    # labels_gray = labels.mean(dim=1)

    # Convert grayscale images to binary masks
    outputs_bin = (outputs > threshold).float()
    labels_bin = (labels > threshold).float()

    outputs_centers = batch_center_of_mass(outputs_bin)
    labels_centers = batch_center_of_mass(labels_bin)

    center_distances = torch.abs(outputs_centers - labels_centers)

    l2 = torch.norm(center_distances, p="fro", dim=1) # frobenius norm

    return l2


def batch_center_of_mass(bin_images, device="cuda"):
    """
    Calculate the center of mass of binary pixels for each image in a mini-batch in a vectorized way.

    Parameters:
    bin_images (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                         representing binary images.

    Returns:
    Tensor: Tensor of shape (B, 2) containing the (x, y) coordinates of the center of mass for each image.
    """
    # Create meshgrid for coordinates
    x_coord = torch.arange(bin_images.shape[3]).to(device)
    y_coord = torch.arange(bin_images.shape[2]).to(device)
    xx, yy = torch.meshgrid(y_coord, x_coord)

    # Calculate the center of mass
    centers_x = (bin_images * xx).sum(dim=(1, 2, 3)) / bin_images.sum(dim=(1, 2, 3))
    centers_y = (bin_images * yy).sum(dim=(1, 2, 3)) / bin_images.sum(dim=(1, 2, 3))

    # Stack the coordinates
    centers = torch.stack((centers_x, centers_y), dim=-1)

    return centers

    # old function
    centers = []
    t = torch.nonzero(bin_images, as_tuple=False) # coods of true pixels
    batch_images = torch.unique(t[:, 0])

    for image in batch_images:
        # Get the rows for the current batch
        batch_rows = t[t[:, 0] == image]

        # Calculate the min and max coordinates
        min_w, max_w = batch_rows[:, 2].min(), batch_rows[:, 2].max()
        min_h, max_h = batch_rows[:, 3].min(), batch_rows[:, 3].max()

        # Calculate the center of the bounding box
        center_w = (min_w + max_w) / 2
        center_h = (min_h + max_h) / 2

        centers.append([center_w.item(), center_h.item()])

    # Convert the list of centers to a tensor
    centers = torch.tensor(centers)

    return centers

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
