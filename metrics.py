import cv2
import numpy as np
import torch
import torchvision
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

    smooth = 1e-5

    # Convert grayscale images to binary masks
    outputs_bin = (outputs > threshold)
    labels_bin = (labels > threshold)

    intersection = (outputs_bin & labels_bin).sum((1, 2, 3))  # Will be zero if Truth=0, Prediction=0 or no overlap
    union = (outputs_bin | labels_bin).sum((1, 2, 3))         # Will be zero if both are 0

    iou = (intersection + smooth) / union + smooth # We smooth our division to avoid 0/0

    return iou


def center_distance_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5, device="cpu"):
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

    outputs_centers = batch_center_of_mass(outputs_bin, device=device)
    labels_centers = batch_center_of_mass(labels_bin, device=device)

    center_distances = torch.abs(outputs_centers - labels_centers)

    l2 = torch.norm(center_distances, p="fro", dim=1) # frobenius norm

    return l2


def batch_center_of_mass(bin_images, device="cpu"):
    """
    Calculate the center of mass of binary pixels for each image in a mini-batch in a vectorized way.

    Parameters:
    bin_images (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                         representing binary images.

    Returns:
    Tensor: Tensor of shape (B, 2) containing the (x, y) coordinates of the center of mass for each image.
    """
    bin_images = bin_images.to(device)

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


def center_shapes(images: torch.Tensor, treshold=0.5):
    """
    Center the shapes in binary images.

    Parameters:
    images (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of binary images all of the same size.

    Returns:
    Tensor: Images with centered shapes.
    """
    imgs = torch.clone(images)
    B, C, H, W = images.shape
    centers = batch_center_of_mass(imgs)
    centers[:,0] = (H-1) / 2 - centers[:,0]
    centers[:,1] = (W-1) / 2 - centers[:,1] 
    centers = torch.round(centers).int()
 
    for idx, image in enumerate(images):
        images[idx,:,:,:] = torchvision.transforms.functional.affine(image, 
                                                        translate = (centers[idx][1], centers[idx][0]),
                                                        angle=0,
                                                        scale=1,
                                                        shear=0)
    return images


def max_diameter_and_angle(image):
    """
    Calculate the maximum diameter and corresponding angle in an image.

    This function accepts both numpy arrays and PyTorch tensors as input. If the input is a PyTorch tensor, it is converted to a numpy array before processing. The image is then converted to grayscale and thresholded to extract the shape. The function finds contours in the thresholded image and calculates the maximum diameter and corresponding angle.

    Parameters:
    image (numpy.ndarray or torch.Tensor): The input image. If a PyTorch tensor is provided, it should have the shape (Channels, Height, Width).

    Returns:
    float: The maximum diameter found in the image.
    list: A list of angles (in degrees) corresponding to the maximum diameter.
    list: A list of positions corresponding to the maximum diameter. Each position is a list of two tuples, where the first tuple represents the x-coordinates and the second tuple represents the y-coordinates.
    """
    # Check if the image is a PyTorch tensor
    if torch.is_tensor(image):
        # Convert the PyTorch tensor to a numpy array
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] > 1:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Threshold the image to get the shape
    _, threshold = cv2.threshold(image, 127, 255, 0)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Initialize maximum diameter to 0
    max_diameter = 0
    angle = []
    pos = []

    # Calculate maximum diameter and corresponding angle
    for cnt in contours:
        for i in range(len(cnt)):
            for j in range(i+1, len(cnt)):
                x1 = cnt[i][0][0]
                y1 = cnt[i][0][1]
                x2 = cnt[j][0][0]
                y2 = cnt[j][0][1]

                diameter = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if diameter >= max_diameter:

                    if diameter > max_diameter:
                        max_diameter = diameter
                        angle = []
                        pos = []

                    angle.append(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    pos.append([(x1, x2), (y1, y2)])

    return max_diameter, angle, pos


def min_angle_distance(list1, list2):
    """
    Calculate the minimum angle distance between two lists of angles.

    This function iterates over each pair of angles (one from each list), and calculates the absolute difference between them.
    If the difference is greater than 180, it considers the shorter distance around the circle (360 degrees).
    It keeps track of the smallest difference encountered, which is returned as the result.

    Parameters:
    list1 (List[float]): The first list of angles (in degrees).
    list2 (List[float]): The second list of angles (in degrees).

    Returns:
    float: The minimum angle distance between the two lists of angles.
    """
    min_distance = 360  # Initialize minimum distance to the maximum possible value
    for angle1 in list1:
        for angle2 in list2:
            # Calculate the absolute difference between the two angles
            diff = abs(angle1 - angle2)
            # If the difference is greater than 180, it is faster to go the other way around the circle
            if diff > 180:
                diff = 360 - diff
            # Update the minimum distance if necessary
            if diff < min_distance:
                min_distance = diff
    return min_distance


def contour_length(image):
    """
    Calculate the total length of all contours in a grayscale image.

    This function loads a grayscale image from the specified path, thresholds it to binary (black and white), finds the contours in the binary image, and then calculates the total length of these contours. The length of a contour is approximated as the sum of the Euclidean distances between consecutive points on the contour.

    Parameters:
    image_path (str): The path to the grayscale image file. The image should have white pixels representing the geometric shape and black pixels representing the background.

    Returns:
    float: The total length of all contours in the image, in pixels.
    """
    if torch.is_tensor(image):
        # Convert the PyTorch tensor to a numpy array
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] > 1:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image: this will make the geometric shape white (255) and the background black (0)
    _, threshold = cv2.threshold(image, 127, 255, 0)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize total length to 0
    total_length = 0

    # Calculate the length of each contour and add it to the total length
    for contour in contours:
        total_length += cv2.arcLength(contour, True)

    return total_length