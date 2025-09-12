from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import requests
import time
import uuid
import json
import logging
import io

# Configure logging with timestamps and structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None  # Starting image URL for editing
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A beautiful sunset over mountains",
                "image_url": "https://example.com/image.jpg"
            }
        }


# --- Config ---
IMAGE_UPLOAD_URL = "https://wallpaperaccess.com/full/1556608.jpg"


def upload_image_to_uguu(image_url: str) -> str:
    """Download image and upload to uguu.se"""
    logger.info(f"Downloading image from: {image_url}")
    
    try:
        # Download the image
        response = requests.get(image_url, headers={"Referer": "https://visualgpt.io"}, timeout=30)
        response.raise_for_status()
        
        # Check if we got image data
        if not response.content:
            raise Exception("Empty image data received")
        
        # Create file object from image data
        image_data = io.BytesIO(response.content)
        
        # Extract filename from URL or use default
        filename = image_url.split("/")[-1]
        if not filename or "." not in filename:
            filename = f"generated_image_{int(time.time())}.jpg"
        
        logger.info(f"⏳ Uploading image to uguu.se as {filename}...")
        
        files = {"files[]": (filename, image_data, "image/jpeg")}
        
        upload_response = requests.post("https://uguu.se/upload", files=files, timeout=60)
        
        if upload_response.status_code != 200:
            logger.error(f"Upload failed with status {upload_response.status_code}")
            raise Exception(f"Upload failed with status {upload_response.status_code}")
        
        # Check if response has content
        if not upload_response.text or not upload_response.text.strip():
            logger.error("Empty response from uguu.se")
            raise Exception("Empty response from uguu.se")
        
        try:
            data = upload_response.json()
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from uguu.se. Response: {upload_response.text[:200]}...")
            raise Exception(f"Invalid JSON response from uguu.se: {str(e)}")
        
        if not data.get("success") or not data.get("files"):
            logger.error(f"Unexpected upload response: {data}")
            raise Exception(f"Unexpected upload response: {data}")
        
        url = data["files"][0].get("url") if data.get("files") and len(data["files"]) > 0 else None
        
        if not url:
            logger.error("No URL returned from upload")
            raise Exception("No URL returned from upload")
        
        logger.info(f"✅ Upload successful: {url}")
        return url
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during upload: {str(e)}")
        raise Exception(f"Network error during upload: {str(e)}")


def generate_cookie():
    anon_user_id = str(uuid.uuid4())  # Generates a new unique ID
    return (
        "_ga=GA1.1.1802416543.1757653998; "
        "_ga_PZ8PZQP57J=GS2.1.s1757653997$o1$g0$t1757653997$j60$l0$h0; "
        f"anonymous_user_id={anon_user_id}; "
        "sbox-guid=MTc1NzY1Mzk5OXw0OTJ8OTMxOTE4NDg0; "
        "crisp-client%2Fsession%2Fe48f416d-75e9-492f-9624-e9a4772aef40=session_8314853f-cfcb-4730-a7cb-9accce3fcce4"
    )


class VisualGPTProvider:
    def __init__(self):
        self.cookie_string = generate_cookie()
        self.headers_step1 = {
            "authority": "visualgpt.io",
            "method": "POST",
            "path": "/api/v1/prediction/handle",
            "scheme": "https",
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json; charset=UTF-8",
            "cookie": self.cookie_string,
            "origin": "https://visualgpt.io",
            "referer": "https://visualgpt.io/ai-models/nano-banana",
            "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Microsoft Edge";v="140"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
        }
        
        self.headers_step2 = {
            "authority": "visualgpt.io",
            "method": "GET",
            "path": "",  # will fill with `get-status?session_id=...`
            "scheme": "https",
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "cookie": self.cookie_string,
            "referer": "https://visualgpt.io/ai-models/nano-banana",
            "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Microsoft Edge";v="140"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
        }

    def submit_prediction(self, prompt, image_url=None):
        url = "https://visualgpt.io/api/v1/prediction/handle"
        
        # Use provided image_url or default
        starting_image = image_url if image_url and image_url.strip() else IMAGE_UPLOAD_URL
        
        payload = {
            "image_urls": [starting_image],
            "type": 61,
            "user_prompt": prompt,
            "sub_type": 2,
            "aspect_ratio": "",
            "num": ""
        }

        logger.info(f"Submitting prediction request for prompt: {prompt[:50]}... with image: {starting_image}")
        try:
            resp = requests.post(url, json=payload, headers=self.headers_step1, timeout=30)
            resp.raise_for_status()
            
            # Check if response has content
            if not resp.text or not resp.text.strip():
                logger.error(f"Empty response from VisualGPT API")
                raise Exception("Empty response from VisualGPT API")
            
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error. Response content: {resp.text[:200]}...")
                raise Exception(f"Invalid JSON response from VisualGPT API: {str(e)}")
            
            if data.get("code") == 100000 and "session_id" in data.get("data", {}):
                session_id = data["data"]["session_id"]
                logger.info(f"Prediction submitted successfully, session_id: {session_id}")
                return session_id
            else:
                logger.error(f"Submission failed: {data.get('message', 'unknown error')}")
                raise Exception(f"Submission failed: {data.get('message', 'unknown error')}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during prediction submission: {str(e)}")
            raise Exception(f"Network error: {str(e)}")

    def poll_status(self, session_id, timeout=120, interval=5):
        url = f"https://visualgpt.io/api/v1/prediction/get-status?session_id={session_id}"
        start = time.time()
        logger.info(f"Starting to poll status for session_id: {session_id}")
        
        while True:
            try:
                headers = self.headers_step2.copy()
                # update path header so it matches (optional but good replication)
                headers["path"] = f"/api/v1/prediction/get-status?session_id={session_id}"
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                
                # Check if response has content
                if not resp.text or not resp.text.strip():
                    logger.error(f"Empty response from status API")
                    raise Exception("Empty response from status API")
                
                try:
                    data = resp.json()
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in status check. Response: {resp.text[:200]}...")
                    raise Exception(f"Invalid JSON response from status API: {str(e)}")

                # if server signals session timeout
                msg = data.get("message", "")
                if "time" in msg.lower() and "out" in msg.lower():
                    logger.error("Session timed out on server side")
                    raise Exception("Session timed out on server side.")

                results = data.get("data", {}).get("results", []) if data.get("data") else []
                if results:
                    result = results[0]
                    status = result.get("status", "")
                    logger.info(f"Current status: {status}")
                    if status == "succeeded":
                        img_url = result.get("url", "")
                        if img_url:
                            logger.info(f"Image generation succeeded: {img_url}")
                            return img_url
                        else:
                            logger.error("Succeeded status but URL is empty")
                            raise Exception("Succeeded status but URL is empty.")
                    elif status == "failed":
                        logger.error("Image generation failed")
                        raise Exception("Generation failed.")
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during status polling: {str(e)}")
                raise Exception(f"Network error during status check: {str(e)}")
                
            # Timeout check
            if time.time() - start > timeout:
                logger.error(f"Timeout waiting for image generation after {timeout}s")
                raise Exception("Script timeout waiting for image generation.")
            time.sleep(interval)

    async def generate_image(self, prompt: str, image_url: str = None) -> str:
        """Generate an image and return the uploaded URL from uguu.se"""
        try:
            # Use default URL if image_url is None or empty
            starting_image = image_url if image_url and image_url.strip() else IMAGE_UPLOAD_URL
            logger.info(f"Starting image generation for prompt: {prompt[:50]}... with starting_image: {starting_image}")
            session_id = self.submit_prediction(prompt, starting_image)
            original_image_url = self.poll_status(session_id)
            
            # Upload to uguu.se and return the new URL
            uploaded_url = upload_image_to_uguu(original_image_url)
            logger.info(f"Image generation and upload completed successfully")
            return uploaded_url
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# --- App Init ---
app = FastAPI(
    title="🍌 Nano Banana Image Generation API", 
    version="1.0.0",
    description="""A powerful image generation API that creates images from text prompts and uploads them to uguu.se.
    
## Features
- Generate images from text descriptions
- Optional starting image for editing
- Automatic upload to uguu.se for easy sharing
- Real-time generation status with structured logging
    
## Usage
1. Use the `/v1/image/generations` endpoint for programmatic access
2. Check `/health` for service status
3. All generated images are automatically uploaded to uguu.se
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)
provider = VisualGPTProvider()


# --- Routes ---
@app.post("/v1/image/generations", 
         summary="Generate Image",
         description="Generate an image based on a text prompt and upload it to uguu.se. Optionally provide a starting image URL for editing.",
         response_description="Returns the uploaded image URL from uguu.se")
async def create_image(request: ImageGenerationRequest):
    """Generate an image based on a text prompt and upload to uguu.se.
    
    - **prompt**: The text description of the image you want to generate
    - **image_url**: Optional starting image URL for editing (uses default if not provided)
    
    Returns the uploaded image URL from uguu.se instead of the original generation URL.
    """
    logger.info(f"Received image generation request: prompt='{request.prompt[:50]}...', image_url='{request.image_url}'")
    
    if not request.prompt or not request.prompt.strip():
        logger.error("Empty prompt provided")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    try:
        uploaded_url = await provider.generate_image(
            request.prompt, 
            request.image_url if request.image_url and request.image_url.strip() else None
        )
        
        created_ts = int(time.time())
        
        response_payload = {
            "created": created_ts,
            "data": [
                {
                    "url": uploaded_url,
                    "revised_prompt": request.prompt
                }
            ]
        }
        
        logger.info(f"Successfully generated and uploaded image, returning uguu.se URL")
        return JSONResponse(content=response_payload)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/", summary="API Info", description="Get basic information about the API")
async def root():
    """Get API information"""
    return {
        "message": "🍌 Nano Banana Image Generation API", 
        "version": "1.0.0",
        "description": "Generate images from text prompts and upload to uguu.se",
        "endpoints": {
            "POST /v1/image/generations": "Generate image from prompt",
            "GET /docs": "Swagger UI documentation",
            "GET /health": "Health check"
        }
    }


@app.get("/health", summary="Health Check", description="Check if the API service is running")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Nano Banana Image Generation API"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Nano Banana Image Generation API server on port 10000")
    uvicorn.run("test:app", host="0.0.0.0", port=10000, reload=True)