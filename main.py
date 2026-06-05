import io
import hashlib
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


class PredictionOutput(BaseModel):
    """
    Pydantic model defining the structure of the API response for image classification results.
    """

    prediction: str = Field(
        ..., description="Human-readable classification outcome string."
    )
    confidence: float = Field(
        ..., description="Model certainty score ranging between 0.0 and 1.0."
    )

    class Config:
        """
        Pydantic model configuration class to provide additional metadata for API documentation.
        """

        json_schema_extra = {
            "example": {"prediction": "Synthetic/Fake", "confidence": 0.9845}
        }


request_history = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiViewNet(view_configuration=MODEL_CONFIGURATION, embed_dim=512).to(device)
trained_model = "models/trained_model.pt"
model.load_state_dict(torch.load(trained_model, map_location=device))
model.eval()


@app.post(
    "/api_post/",
    response_model=PredictionOutput,
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
    """
    API endpoint for classifying uploaded images as AI-generated or natural.
    Parameters:
        - file: An image file uploaded by the user. Supported formats include .jpg, .jpeg, and .png.
    Returns:
        - A JSON object containing the classification result and confidence score.
    Error Handling:
        - 400 Bad Request: If no file is uploaded, if the file cannot be decoded as an image, or if the file is corrupted.
        - 415 Unsupported Media Type: If the uploaded file is not in a supported image format.
        - 422 Unprocessable Entity: If the image does not meet minimum size requirements (less than 256x256 pixels).
    """
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

    # check if the image is passing minimum size requirements
    width, height = img.size
    if width < PATCH_SIZE or height < PATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Image size is too small ({width}x{height}px). Minimum size required is {PATCH_SIZE}x{PATCH_SIZE}px.",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = MultiViewNet(view_configuration=MODEL_CONFIGURATION, embed_dim=512).to(
        device
    )
    weights_checkpoint = "checkpoints/multiview_model_epoch_11.pt"
    model.load_state_dict(torch.load(weights_checkpoint, map_location=device))
    model.eval()
    preprocessed_input = preprocess(img, MODEL_CONFIGURATION, PATCH_SIZE)
    input_tensors = {
        k: v.unsqueeze(0).to(device) for k, v in preprocessed_input.items()
    }

    with torch.no_grad():
        outputs = model(input_tensors)
        fusion_probs = F.softmax(outputs["fusion"], dim=1).squeeze(0)
        confidence = fusion_probs[1].item()

        # if we want to return confidence per view at some point
        view_details = {}
        for branch_name, branch_logits in outputs["branches"].items():
            branch_probs = F.softmax(branch_logits, dim=1).squeeze(0)
            view_details[branch_name] = round(branch_probs[1].item(), 4)

    prediction = "Synthetic/Fake" if confidence > 0.5 else "Authentic/Real"
    print(prediction, confidence, view_details)
    prediction_result = {"prediction": prediction, "confidence": confidence}

    request_history[file_hash] = prediction_result

    return prediction_result


def extract_patch(image: Image.Image, patch_size=PATCH_SIZE) -> Image.Image:
    """
    Extracts a central patch from the input image.
    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of the patch to extract.
    Returns:
        PIL.Image.Image: The extracted patch.
    """
    width, height = image.size
    left = (width - patch_size) // 2
    top = (height - patch_size) // 2
    right = left + patch_size
    bottom = top + patch_size

    return image.crop((left, top, right, bottom))


def convert_image(image: Image.Image):
    """Converts the input image to JPEG format if it is not already in that format, while preserving visual fidelity as much as possible.
    This function also handles images with transparency by compositing them onto a white background before conversion.
    Args:        image (PIL.Image.Image): The input image to be converted.
    Returns:        PIL.Image.Image: The converted image in JPEG format.
    """
    if image.format == "JPEG":
        return image

    # check for transpatency and convert to RGB with white background if needed
    if image.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(
            image, mask=image.split()[-1] if image.mode == "RGBA" else None
        )
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG", quality=95, optimize=True)
    buffer.seek(0)
    jpeg_image = Image.open(buffer)
    jpeg_image.load()

    return jpeg_image


def preprocess(
    image_: Image.Image,
    model_configuration: dict = MODEL_CONFIGURATION,
    patch_size: int = PATCH_SIZE,
) -> dict:
    """
    Preprocesses the input image according to the specified view configuration.
    Parameters:
        - image: A PIL Image object to be preprocessed.
        - model_configuration: A dictionary containing the configuration for each view, including the preprocessor and expected input shape.
        - patch_size: The size of the central patch to be extracted from the image for processing.
    Returns:
        - A dictionary containing the preprocessed views ready for model inference.
    """
    image_ = convert_image(image_)
    patch = extract_patch(image_, patch_size)

    view_outputs = {}
    preprocessors = {
        view_name: config_["preprocessor"]
        for view_name, config_ in model_configuration.items()
    }

    for view_name, preprocessor in preprocessors.items():
        raw_output = preprocessor(patch)

        if not isinstance(raw_output, torch.Tensor):
            tensor_output = torch.from_numpy(raw_output).float()
        else:
            tensor_output = raw_output.float()

        view_outputs[view_name] = tensor_output.detach()
    return view_outputs
