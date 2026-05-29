import io
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from MVL_AI_Classifier.constants import PATCH_SIZE

app = FastAPI(
    title="Multi-view AI Detection API",
    description="Advanced API pipeline designed to analyze images for synthetic or AI-generated manipulation signatures.",
)


# Placeholder for model loading function
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


request_history = {}
# Replace with actual model loading code, e.g., model = load_model("path_to_model")
model = None


@app.post(
    "/api_post/",
    response_model=PredictionOutput,  # <-- Tells FastAPI docs what data types to expect
    status_code=200,
)
async def inf(file: UploadFile = File(None)):
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

    # Checking if the image is passing minimum size requirements
    width, height = img.size
    if width < PATCH_SIZE or height < PATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Image size is too small ({width}x{height}px). Minimum size required is {PATCH_SIZE}x{PATCH_SIZE}px.",
        )

    # Placeholder for actual prediction code, e.g., prediction, confidence = model.predict(img)
    prediction, confidence = model.predict(img)

    prediction_result = {"prediction": prediction, "confidence": confidence}

    request_history[file_hash] = prediction_result

    return prediction_result
