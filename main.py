import io
import hashlib
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from MVL_AI_Classifier.constants import PATCH_SIZE
from MVL_AI_Classifier.model_configuration import MODEL_CONFIGURATION
from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet

app = FastAPI(
    title="Multi-view AI Detection API",
    description="""
    This enterprise-grade REST API hosts a **4-branch forensic neural network** optimized to detect synthetic, AI-generated, or structurally manipulated digital media.
    
    ### Core Capabilities
    *  **Noise Residual Extraction:** Evaluates camera sensor pattern inconsistencies.
    *  **Frequency Domain Analysis:** Cross-references Discrete Cosine (DCT) and Amplitude Spectrum (APS) profiles.
    *  **Texture Mapping:** Examines micro-textures using Gray-Level Co-occurrence Matrices (GLCM).
    
    ### Operational Constraints
    Input payloads must be structural valid `.jpg`, `.jpeg`, or `.png` files satisfying a minimum spatial bounding canvas of **256x256 pixels**.
    """,
    version="1.0.0",
)

# Initialize the multi-view model and load pre-trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiViewNet(view_configuration=MODEL_CONFIGURATION, embed_dim=512).to(device)
trained_model = "models/trained_model.pt"
model.load_state_dict(torch.load(trained_model, map_location=device))
model.eval()


class PatchEvaluation(BaseModel):
    """
    Represents the evaluation result for a single patch within the image grid.
    """

    box: List[int] = Field(
        ...,
        description="Bounding box coordinates ordered as [left, top, right, bottom].",
    )
    prediction: str = Field(
        ..., description="Classification outcome string for this patch."
    )
    confidence: float = Field(
        ...,
        description="Model certainty score for this patch ranging between 0.0 and 1.0.",
    )


class GridPredictionOutput(BaseModel):
    """
    Represents the structured output of the API containing a list of patch evaluations for the entire image grid.
    """

    patch_evaluations: List[PatchEvaluation] = Field(
        ..., description="A collection of matrix grid patch evaluation profiles."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patch_evaluations": [
                    {
                        "box": [112, 112, 368, 368],
                        "prediction": "Synthetic/Fake",
                        "confidence": 0.9845,
                    }
                ]
            }
        }


request_history = {}


# Root endpoint for health check and basic API information
@app.get("/")
def read_root():
    return {"status": "API is running", "docs_url": "/docs"}


# Endpoint for processing uploaded images and returning patch-based forensic evaluations
@app.post(
    "/api_post/",
    response_model=GridPredictionOutput,
    summary="Evaluate an image file for AI-generation signatures",
    description="Accepts an image file upload and returns a classification result indicating whether the image is",
    response_description="A structured JSON footprint detailing forensic evaluation results.",
    status_code=200,
    responses={
        400: {"description": "Bad Request (Missing file or corrupted data)"},
        415: {"description": "Unsupported Media Type (Not a JPG/PNG)"},
        422: {
            "description": "Unprocessable Entity (Image canvas is smaller than 256x256px)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Image size is too small (120x120px). Minimum size required is 256x256px."
                    }
                }
            },
        },
    },
)
async def inf(
    file: UploadFile = File(..., description="A valid .jpg, .jpeg, or .png image file"),
):
    """""
    API endpoint for classifying uploaded images as AI-generated or natural.
    Args:
        - file: An image file uploaded by the user. Supported formats include .jpg, .jpeg, and .png.
    Returns:
        - A JSON object containing a list of patch evaluations, where each evaluation includes the bounding box coordinates,
        the prediction label (either "Synthetic/Fake" or "Authentic/Real"), and the confidence score for that prediction.
    Error Handling:
        - 400 Bad Request: If no file is uploaded, if the file cannot be decoded as an image, or if the file is corrupted.
        - 415 Unsupported Media Type: If the uploaded file is not in a supported image format.
        - 422 Unprocessable Entity: If the image does not meet minimum size requirements (less than 256x256 pixels).
    """ ""
    # Checking if there is a file uploaded
    if file is None or file.filename is None:
        raise HTTPException(
            status_code=400, detail="No file was uploaded or file metadata is missing."
        )

    # Checking if the file is a supported file type
    if not (
        file.filename.endswith(".jpg")
        or file.filename.endswith(".png")
        or file.filename.endswith(".jpeg")
    ):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please use '.jpg' or '.png'.",
        )

    contents = await file.read()

    # Saving the fingerprint of the file for request history
    file_hash = hashlib.md5(contents).hexdigest()

    # Checking ig the file_hash is known
    if file_hash in request_history:
        return request_history[file_hash]

    # Checking for the integrity of the file why trying to read the image
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image files. File may be corrupted.",
        )

    # Checking if the image is passing minimum size requirements
    width, height = img.size
    if width < PATCH_SIZE or height < PATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Image size is too small ({width}x{height}px). Minimum size required is {PATCH_SIZE}x{PATCH_SIZE}px.",
        )

    # Convert the file system image array structure
    img_converted = convert_image(img)

    # Generate complete symmetrical spatial patch boundaries expanding out from the center
    patches_coords = generate_center_out_patches(width, height, PATCH_SIZE)

    patch_results = []

    # Process each individual patch layer through the multi-view analyzer
    for left, top, right, bottom in patches_coords:
        patch = img_converted.crop((left, top, right, bottom))

        preprocessed_input = preprocess_patch(patch, MODEL_CONFIGURATION)
        input_tensors = {
            k: v.unsqueeze(0).to(device) for k, v in preprocessed_input.items()
        }

        with torch.no_grad():
            outputs = model(input_tensors)
            fusion_probs = F.softmax(outputs["fusion"], dim=1).squeeze(0)
            confidence = fusion_probs[1].item()

            # Logging structural details per branch component internally
            view_details = {}
            for branch_name, branch_logits in outputs["branches"].items():
                branch_probs = F.softmax(branch_logits, dim=1).squeeze(0)
                view_details[branch_name] = round(branch_probs[1].item(), 4)

        # Determine the final prediction label based on the confidence score
        prediction = "Synthetic/Fake" if confidence > 0.5 else "Authentic/Real"
        print(
            f"Slice Window [{left}, {top}, {right}, {bottom}] -> Verdict: {prediction} | Conf: {confidence:.4f} | Branches: {view_details}"
        )

        # Append the evaluation result for this patch to the results list
        patch_results.append(
            {
                "box": [left, top, right, bottom],
                "prediction": prediction,
                "confidence": confidence,
            }
        )

    # Compile the final structured output and store it in the request history for potential future reference
    prediction_result = {"patch_evaluations": patch_results}
    request_history[file_hash] = prediction_result

    return prediction_result


def generate_center_out_patches(
    width: int, height: int, patch_size: int
) -> List[List[int]]:
    """
    Generates a list of bounding box coordinates for patches that are symmetrically distributed around the center of the image.
    The function calculates the center point of the image and then expands outwards in both horizontal and
    vertical directions to create a grid of patches that cover the entire image canvas.
    Args:
        - width: The width of the image in pixels.
        - height: The height of the image in pixels.
        - patch_size: The size of each square patch in pixels (e.g., 256).
    Returns:
        - A list of bounding box coordinates for each patch, where each bounding box is represented as [left, top, right, bottom].
    """
    mid_x, mid_y = width // 2, height // 2

    # Calculate step sequences expanding outwards along the horizontal X-axis
    x_steps = [mid_x - patch_size // 2]
    while x_steps[-1] + patch_size + patch_size <= width:
        x_steps.append(x_steps[-1] + patch_size)
    while x_steps[0] - patch_size >= 0:
        x_steps.insert(0, x_steps[0] - patch_size)

    # Calculate step sequences expanding outwards along the vertical Y-axis
    y_steps = [mid_y - patch_size // 2]
    while y_steps[-1] + patch_size + patch_size <= height:
        y_steps.append(y_steps[-1] + patch_size)
    while y_steps[0] - patch_size >= 0:
        y_steps.insert(0, y_steps[0] - patch_size)

    # Perform dot cross production coordinate matching to map the complete window grid
    boxes = []
    for x in x_steps:
        for y in y_steps:
            boxes.append([x, y, x + patch_size, y + patch_size])

    return boxes


def convert_image(image: Image.Image) -> Image.Image:
    """
    Converts the input image to JPEG format if it is not already in that format, while preserving visual fidelity as much as possible.
    This function also handles images with transparency by compositing them onto a white background before conversion.
    Args:
        - image: A PIL Image object that may be in various formats (e.g., PNG with transparency, JPEG, etc.).
    Returns:
        - A PIL Image object in JPEG format, suitable for further processing by the model.
    """
    if image.format == "JPEG":
        return image

    # Handle images with transparency by compositing them onto a white background
    if image.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(
            image, mask=image.split()[-1] if image.mode == "RGBA" else None
        )
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Convert the image to JPEG format in-memory to preserve visual fidelity while
    # ensuring compatibility with the model's expected input format
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    buffer.seek(0)
    jpeg_image = Image.open(buffer)
    jpeg_image.load()

    return jpeg_image


def preprocess_patch(
    patch: Image.Image, model_configuration: dict = MODEL_CONFIGURATION
) -> dict:
    """
    Preprocesses a given image patch using the specified model configuration for each view branch.
    This function applies the appropriate preprocessing steps for each view as defined in the model configuration, ensuring that the input data is correctly formatted and normalized for the multi-view model.
    Args:
        - patch: A PIL Image object representing the image patch to be preprocessed.
        - model_configuration: A dictionary containing the configuration for each view branch, including the specific preprocessor function to apply for that view.
    Returns:
        - A dictionary where the keys are the view names and the values are the corresponding preprocessed
    """
    # Initialize a dictionary to hold the preprocessed outputs for each view branch
    view_outputs = {}
    preprocessors = {
        view_name: config_["preprocessor"]
        for view_name, config_ in model_configuration.items()
    }

    # Apply the appropriate preprocessing function for each view branch as defined in the model configuration
    for view_name, preprocessor in preprocessors.items():
        raw_output = preprocessor(patch)

        if not isinstance(raw_output, torch.Tensor):
            tensor_output = torch.from_numpy(raw_output).float()
        else:
            tensor_output = raw_output.float()

        view_outputs[view_name] = tensor_output.detach()
    return view_outputs
