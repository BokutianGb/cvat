import json
import base64
import io
import os
import numpy as np
import onnxruntime
from PIL import Image
import cv2

# ======================================================================================
# CONFIGURATION
# This part is executed once when the function is initialized.
# ======================================================================================

# The model path is determined by an environment variable, defaulting to 'best.onnx'.
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")

# Initialize the ONNX runtime session. It's recommended to use CPU for broad compatibility,
# but you can configure it for 'CUDAExecutionProvider' if a GPU is available in the deployment environment.
providers = ["CPUExecutionProvider"] 
session = onnxruntime.InferenceSession(MODEL_PATH, providers=providers)

# Retrieve model input details (name and shape) which are crucial for preprocessing.
model_inputs = session.get_inputs()
input_name = model_inputs[0].name
input_shape = model_inputs[0].shape  # e.g., [1, 3, 640, 640]

# Retrieve model output names, required to fetch the results after inference.
model_outputs = session.get_outputs()
output_names = [output.name for output in model_outputs]

def preprocess(image: Image.Image, target_shape: tuple):
    """
    Prepares an image for the YOLOv8/v11 segmentation model.

    The process involves:
    1.  Calculating the correct scale to resize the image while maintaining aspect ratio.
    2.  Resizing the image.
    3.  Padding the image to match the model's square input dimensions (e.g., 640x640).
        Ultralytics models typically use a specific grey color for padding.
    4.  Converting the image from a Pillow object to a NumPy array, normalizing pixel
        values to the [0, 1] range.
    5.  Transposing the array from HWC (Height, Width, Channel) to CHW format.
    6.  Adding a batch dimension to create the final 4D tensor.

    Args:
        image: The input PIL.Image object.
        target_shape: The model's expected input shape (e.g., [1, 3, 640, 640]).

    Returns:
        A tuple containing the processed input tensor and a tuple with scaling/padding
        information needed for postprocessing.
    """
    img_w, img_h = image.size
    in_h, in_w = target_shape[2:]

    # Calculate scaling factor to fit the image within the target dimensions.
    scale = min(in_w / img_w, in_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    
    # Resize image.
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create a grey canvas and paste the resized image onto it.
    padded_image = Image.new("RGB", (in_w, in_h), (114, 114, 114))
    padded_image.paste(image_resized, (0, 0))

    # Convert to NumPy array and normalize.
    input_tensor = np.array(padded_image, dtype=np.float32) / 255.0
    # Transpose from HWC to CHW and add batch dimension.
    input_tensor = np.expand_dims(input_tensor.transpose(2, 0, 1), 0)
    
    return input_tensor, (scale, new_w, new_h, orig_w, orig_h)

def postprocess(outputs, preprocess_info):
    """
    Converts raw model outputs into CVAT-compatible polygon annotations.

    YOLOv8 segmentation models have two main outputs:
    1.  A detection tensor with bounding boxes, confidences, and mask coefficients.
    2.  A tensor of "prototype" masks.

    This function performs the following steps:
    1.  Filters out low-confidence detections.
    2.  For each valid detection, it combines the mask coefficients with the prototype
        masks to generate a final, instance-specific mask.
    3.  Scales this mask from the model's output resolution to the original image size.
    4.  Applies a threshold to create a binary mask.
    5.  Uses OpenCV to find the external contour of the binary mask.
    6.  Formats the contour points into a flat list `[x1, y1, x2, y2, ...]` for CVAT.

    Args:
        outputs: A list of NumPy arrays from the ONNX session.run().
        preprocess_info: A tuple containing scaling/padding info from preprocess().

    Returns:
        A list of dictionaries, where each dictionary represents a polygon annotation
        in the format required by CVAT.
    """
    scale, new_w, new_h, orig_w, orig_h = preprocess_info
    
    # Unpack the two outputs from the model.
    # Detections tensor shape: [1, num_classes + 4_bbox + num_mask_coeffs, num_proposals]
    # For a single class and 32 mask coeffs: [1, 1+4+32, 8400] -> [1, 37, 8400]
    # We transpose it for easier processing.
    detections = outputs[0][0].T  # Shape: [8400, 37]
    # Prototype masks tensor shape: [1, num_mask_coeffs, mask_h, mask_w]
    proto_masks = outputs[1][0]  # Shape: [32, 160, 160]

    # The class confidence for a single-class model is at index 4.
    # Format: [cx, cy, w, h, confidence, mask_coeff_1, ..., mask_coeff_32]
    confidence_threshold = 0.5
    confident_detections = detections[detections[:, 4] > confidence_threshold]

    if confident_detections.shape[0] == 0:
        return []

    scores = confident_detections[:, 4]
    # Bounding boxes are in center-x, center-y, width, height format.
    boxes = confident_detections[:, :4]
    # Mask coefficients are the remaining columns.
    mask_coeffs = confident_detections[:, 5:]

    cvat_results = []
    for i in range(confident_detections.shape[0]):
        # Combine mask coefficients with prototype masks via matrix multiplication.
        instance_mask = (mask_coeffs[i] @ proto_masks.reshape(proto_masks.shape[0], -1))
        instance_mask = instance_mask.reshape(proto_masks.shape[1], proto_masks.shape[2]) # Back to [160, 160]

        # Apply sigmoid to get probabilities.
        instance_mask = 1 / (1 + np.exp(-instance_mask))

        # Upscale the mask from model's output size (e.g., 160x160) to the padded input size (e.g., 640x640).
        in_h, in_w = input_shape[2:]
        upscaled_mask = cv2.resize(instance_mask, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        # Remove the padding area from the mask to match the resized image.
        mask_resized_to_image = upscaled_mask[:new_h, :new_w]
        
        # Scale the mask to the original image dimensions.
        mask_original_size = cv2.resize(mask_resized_to_image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Binarize the final mask.
        binary_mask = (mask_original_size > confidence_threshold).astype(np.uint8)
        
        if np.sum(binary_mask) == 0:
            continue

        # Find contours in the binary mask.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue

        # Convert the largest contour into the flat list format required by CVAT.
        contour = max(contours, key=cv2.contourArea)
        if contour.size < 6: # A valid polygon needs at least 3 points.
            continue
            
        points = contour.flatten().tolist()
        
        cvat_results.append({
            "label": "volleyball",  # IMPORTANT: This must match the label name in your CVAT project.
            "points": points,
            "type": "polygon",
            "confidence": str(scores[i]), # You can optionally send the confidence score.
        })
        
    return cvat_results

def handler(context, event):
    """
    The main serverless function handler called by Nuclio.
    """
    try:
        # Load the JSON body from the event.
        body = json.loads(event.body)
        # Decode the base64 encoded image sent by CVAT.
        image_data = base64.b64decode(body["image"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image for the model.
        input_tensor, preprocess_info = preprocess(image, input_shape)

        # Perform inference.
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # Postprocess the results to get polygons.
        cvat_results = postprocess(outputs, preprocess_info)

        # Return the results as a JSON response.
        return context.Response(body=json.dumps(cvat_results),
                                headers={},
                                content_type='application/json',
                                status_code=200)
    except Exception as e:
        # Log any exceptions and return an error code to CVAT.
        context.logger.error(f"Unhandled exception: {e}", exc_info=True)
        return context.Response(body=f"Error processing image: {e}",
                                headers={},
                                content_type='text/plain',
                                status_code=500) 