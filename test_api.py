import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import cv2
import logging
from PIL import Image, ImageDraw, ImageFont
from PIL import UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from typing import Optional
from datetime import datetime
import google.generativeai as genai
import json
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import pandas as pd
from fuzzywuzzy import fuzz, process
import sqlite3
from passlib.context import CryptContext
from pydantic import BaseModel

# Pydantic model for repair shops request
class RepairShopRequest(BaseModel):
    city: str
    insurance: Optional[str] = None

# Verify uvicorn is available
try:
    import uvicorn
except ImportError as e:
    logging.error(f"Failed to import uvicorn: {e}. Please install uvicorn using 'pip install uvicorn'.")
    raise ImportError(f"Failed to import uvicorn: {e}. Please install uvicorn using 'pip install uvicorn'.")

# Define combined_loss function
def combined_loss(y_true, y_pred, alpha=0.5):
    cce_loss = categorical_crossentropy(y_true, y_pred)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)
    return alpha * cce_loss + (1 - alpha) * dice_loss

# Load environment variables
load_dotenv()

# Apply nest_asyncio
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    filename=r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\fastapi_log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Car Damage Detection API",
    description="An API for detecting car damages using a pre-trained PSPNet model.",
    version="1.2.5",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PSPNet model
try:
    model = load_model(
        r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\output\models\pspnet_car_damage_subset.h5",
        custom_objects={'combined_loss': combined_loss}
    )
    logging.info("PSPNet model loaded successfully at startup")
except Exception as e:
    logging.error(f"Failed to load PSPNet model: {e}")
    raise RuntimeError(f"Failed to load PSPNet model: {e}")

# Configure Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY environment variable not set")
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    logging.info("Gemini API configured successfully")
except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}")
    raise RuntimeError(f"Failed to configure Gemini API: {e}")

# Load repair parts CSV
try:
    repair_parts_df = pd.read_csv(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\data\repair_parts.csv")
    logging.info(f"Repair parts data loaded successfully from CSV. Columns: {list(repair_parts_df.columns)}")
except Exception as e:
    logging.error(f"Failed to load repair parts CSV: {e}")
    raise RuntimeError(f"Failed to load repair parts CSV: {e}")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# SQLite database setup
def init_db():
    conn = sqlite3.connect(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, hashed_password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, image_path TEXT, 
                  result TEXT, timestamp TEXT,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    conn.commit()
    conn.close()

init_db()

# Preprocess image
def preprocess_image(image: Image.Image, img_size: tuple = (240, 240)) -> tuple:
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR) / 255.0
    image = np.expand_dims(image, axis=0)
    return image, original_image

# Post-process mask
def post_process_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask = np.argmax(mask, axis=-1)
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = (mask > 127).astype(np.uint8) * 255
    return mask

# Estimate damage severity
def estimate_damage_severity(mask: np.ndarray) -> str:
    total_pixels = mask.size
    damaged_pixels = np.sum(mask == 255)
    damage_ratio = damaged_pixels / total_pixels
    logging.info(f"Damage ratio: {damage_ratio:.4f}")
    if damage_ratio < 0.1:
        return "Low"
    elif damage_ratio < 0.3:
        return "Medium"
    else:
        return "High"

# Map text to part names
def map_to_part_name(text: str, available_parts: list, threshold: int = 70) -> list:
    text = text.lower().replace('panel', 'shell').replace('tailgate', 'quarter panel')
    matches = process.extract(text, available_parts, scorer=fuzz.partial_ratio, limit=5)
    matched_parts = [match[0] for match in matches if match[1] >= threshold]
    logging.info(f"Fuzzy matching for '{text}': {[(m[0], m[1]) for m in matches]}")
    return matched_parts

# Infer parts from mask (heuristic-based)
def infer_parts_from_mask(mask: np.ndarray, available_parts: list) -> list:
    height, width = mask.shape
    damaged_pixels = np.where(mask == 255)
    if not damaged_pixels[0].size:
        return []
    y_center = np.mean(damaged_pixels[0]) / height  # 0 (top) to 1 (bottom)
    x_center = np.mean(damaged_pixels[1]) / width   # 0 (left) to 1 (right)
    inferred_parts = []
    if y_center > 0.7:  # Bottom of image
        if x_center > 0.3 and x_center < 0.7:  # Center
            inferred_parts.extend([p for p in available_parts if 'rear bumper' in p.lower()])
    elif y_center < 0.3:  # Top of image
        if x_center > 0.3 and x_center < 0.7:
            inferred_parts.extend([p for p in available_parts if 'front bumper' in p.lower()])
    logging.info(f"Mask-based inference: y_center={y_center:.2f}, x_center={x_center:.2f}, parts={inferred_parts}")
    return inferred_parts

# Generate composite image
def generate_composite(original: np.ndarray, mask: np.ndarray, img_size: tuple = (240, 240)) -> np.ndarray:
    original = cv2.resize(original, img_size, interpolation=cv2.INTER_LINEAR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    color_mask = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    damage_regions = mask == 255
    non_damage_regions = mask == 0
    color_mask[damage_regions, 2] = 255
    color_mask[non_damage_regions, 1] = 255
    alpha = 0.5
    composite = cv2.addWeighted(original, 1 - alpha, color_mask, alpha, 0.0)
    return composite

# Generate highlighted image
def generate_highlighted_image(image: Image.Image, mask: np.ndarray, detected_parts: list) -> Image.Image:
    image_array = np.array(image)
    output_size = (512, 512)
    image_array = cv2.resize(image_array, output_size, interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.GaussianBlur(mask_resized.astype(float), (5, 5), 0)
    mask_resized = (mask_resized > 127).astype(np.uint8) * 255
    glow = cv2.GaussianBlur(mask_resized, (15, 15), 10)
    glow = np.clip(glow, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = image_array.copy()
    alpha = 0.25
    overlay = np.zeros_like(highlighted)
    overlay[mask_resized == 255] = [200, 0, 0]
    overlay[glow > 50] = [255, 50, 50]
    highlighted = cv2.addWeighted(highlighted, 1 - alpha, overlay, alpha, 0.0)
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 1)
    pil_highlighted = Image.fromarray(highlighted)
    draw = ImageDraw.Draw(pil_highlighted)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    text = f"Damaged: {', '.join(detected_parts[:3])}"
    draw.rectangle((5, 5, 300, 30), fill=(0, 0, 0, 128))
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return pil_highlighted

# Validate Gemini output
def validate_genai_output(mask: np.ndarray, highlighted_image: Image.Image) -> bool:
    highlighted_array = np.array(highlighted_image)
    red_channel = (highlighted_array[:, :, 0] >= 200) & (highlighted_array[:, :, 1] <= 50) & (highlighted_array[:, :, 2] <= 50)
    genai_mask = red_channel.astype(np.uint8) * 255
    genai_mask_resized = cv2.resize(genai_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = np.logical_and(mask, genai_mask_resized).sum()
    union = np.logical_or(mask, genai_mask_resized).sum()
    iou = intersection / union if union > 0 else 0
    logging.info(f"Gemini IoU: {iou:.4f}")
    return bool(iou > 0.7)

# Authentication endpoints
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = pwd_context.hash(password)
    c.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", 
              (username, hashed_password))
    conn.commit()
    conn.close()
    logging.info(f"User registered: {username}")
    return {"message": "User registered"}

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\users.db")
    c = conn.cursor()
    c.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if not result or not pwd_context.verify(password, result[0]):
        logging.error(f"Login failed for {username}: Invalid credentials")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logging.info(f"User logged in: {username}")
    return {"message": "Logged in", "username": username}

# History endpoint
@app.get("/history")
async def get_history(username: str):
    conn = sqlite3.connect(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\users.db")
    c = conn.cursor()
    c.execute("SELECT image_path, result, timestamp FROM history WHERE username = ?", 
              (username,))
    history = [{"image_path": row[0], "result": row[1], "timestamp": row[2]} 
               for row in c.fetchall()]
    conn.close()
    logging.info(f"Retrieved history for {username}: {len(history)} records")
    return history

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Car Damage Detection API",
        "description": "Upload an image to detect car damages or process claims.",
        "docs": "/docs",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.3, ge=0.0, le=1.0),
    return_raw: bool = Query(False)
):
    try:
        if not file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
            logging.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")
        contents = await file.read()
        if not contents:
            logging.error(f"Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        logging.info(f"File received: {file.filename}, size: {len(contents)} bytes, MIME: {file.content_type}")
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image.verify()
        except UnidentifiedImageError as e:
            logging.error(f"Invalid image file: {file.filename}, error: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file: cannot identify image format")
        except Exception as e:
            logging.error(f"Image validation failed for {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Image validation failed: {str(e)}")
        
        image_array, original = preprocess_image(image)
        prediction = model.predict(image_array, verbose=0)[0]
        prediction_mean = float(np.mean(prediction))
        prediction_max = float(np.max(prediction))
        logging.info(f"Prediction stats for {file.filename}: mean={prediction_mean:.4f}, max={prediction_max:.4f}, threshold={threshold}")
        if return_raw:
            raw_mask = post_process_mask(prediction)
            output_path = os.path.join(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\visuals", f"raw_mask_{os.path.basename(file.filename)}.png")
            cv2.imwrite(output_path, raw_mask)
            _, buffer = cv2.imencode('.png', raw_mask)
            output_filename = "raw_prediction.png"
        else:
            predicted_mask = (prediction > threshold).astype(np.float32)
            predicted_mask = post_process_mask(predicted_mask)
            composite = generate_composite(original, predicted_mask)
            _, buffer = cv2.imencode('.png', cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            output_filename = "composite.png"
        buffer = io.BytesIO(buffer.tobytes())
        logging.info(f"Prediction successful for file: {file.filename}, return_raw={return_raw}")
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
    except Exception as e:
        logging.error(f"Prediction failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/process_claim")
async def process_claim(
    file: UploadFile = File(...),
    description: str = Form(...),
    vehicle_type: str = Form("sedan", enum=["hatchback", "sedan", "suv", "muv", "premium"]),
    username: str = Form("anonymous"),
    auto_approval: Optional[str] = Form(None),
    estimate: Optional[str] = Form(None)
):
    try:
        if not file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
            logging.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")
        
        contents = await file.read()
        if not contents:
            logging.error(f"Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        logging.info(f"File received: {file.filename}, size: {len(contents)} bytes, MIME: {file.content_type}, first_bytes: {contents[:10].hex()}")
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image.verify()
        except UnidentifiedImageError as e:
            logging.error(f"Invalid image file: {file.filename}, error: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file: cannot identify image format")
        except Exception as e:
            logging.error(f"Image validation failed for {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Image validation failed: {str(e)}")
        
        image_array, original = preprocess_image(image)
        prediction = model.predict(image_array, verbose=0)[0]
        predicted_mask = (prediction > 0.3).astype(np.float32)
        mask = post_process_mask(predicted_mask)
        
        output_path = os.path.join(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\visuals", f"processed_mask_{os.path.basename(file.filename)}.png")
        cv2.imwrite(output_path, mask)
        
        severity = estimate_damage_severity(mask)

        # Infer parts from PSPNet mask
        available_parts = repair_parts_df['Part Name'].tolist()
        mask_parts = infer_parts_from_mask(mask, available_parts)
        logging.info(f"PSPNet inferred parts: {mask_parts}")

        # Gemini API for image-based part detection
        prompt = (
            f"Identify damaged car parts in the provided image. "
            f"Use only part names from this list: {', '.join(available_parts)}. "
            "Return only the part names, separated by commas. "
            "If no parts are detected, return 'Unknown'."
        )
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        try:
            response = gemini_model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_bytes}
            ])
            detected_parts_text = response.text.strip()
            logging.info(f"Gemini raw detected parts: {detected_parts_text}")
        except Exception as e:
            logging.error(f"Gemini API failed for {file.filename}: {e}")
            detected_parts_text = "Unknown"

        detected_parts = [p.strip() for p in detected_parts_text.split(',') if p.strip()]
        if not detected_parts or detected_parts == ['Unknown']:
            detected_parts = mask_parts
            logging.info(f"Falling back to PSPNet parts: {detected_parts}")

        # Get description-based parts
        description_parts = map_to_part_name(description, available_parts)
        logging.info(f"Description-based parts: {description_parts}")

        # Validate description against image-based detection
        if description_parts and detected_parts:
            description_lower = [p.lower() for p in description_parts]
            detected_lower = [p.lower() for p in detected_parts]
            if not any(p in detected_lower for p in description_lower):
                logging.error(f"Mismatch: Description parts {description_parts} not in detected parts {detected_parts}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Description mismatch: '{description}' does not match detected damage ({', '.join(detected_parts)})"
                )

        # Combine parts, prioritizing image-based detection
        detected_parts = list(set(detected_parts + (mask_parts if not detected_parts else [])))
        if not detected_parts:
            logging.warning(f"No parts detected for description: {description}")
            detected_parts = []

        if 'right' in description.lower():
            detected_parts = [p for p in detected_parts if 'right' in p.lower() or 'rear' in p.lower()]
        if 'tailgate' in description.lower() or 'quarter' in description.lower():
            detected_parts = list(set(detected_parts + [p for p in available_parts if 'quarter' in p.lower() and 'rear' in p.lower()]))

        vehicle_type_map = {
            'hatchback': 'Hatchback',
            'sedan': 'Sedan',
            'suv': 'SUV',
            'muv': 'MUV/MPV',
            'premium': 'Premium'
        }
        vehicle_prefix = vehicle_type_map.get(vehicle_type.lower(), 'Sedan')
        
        price_col = f"{vehicle_prefix} Price (INR)"
        paint_col = f"{vehicle_prefix} Paint Cost (INR)"
        fitment_col = f"{vehicle_prefix} Fitment Hrs"
        paint_hrs_col = f"{vehicle_prefix} Paint Hrs"

        df_columns = repair_parts_df.columns
        for col in [price_col, paint_col, fitment_col, paint_hrs_col]:
            if col not in df_columns:
                logging.warning(f"Column '{col}' not found in CSV, using default values")
                repair_parts_df[col] = 5000.0 if 'Price' in col else (1000.0 if 'Paint Cost' in col else 1.0)

        parts_details = []
        total_cost = 0.0
        for part in detected_parts:
            part_data = repair_parts_df[repair_parts_df['Part Name'].str.lower() == part.lower()]
            if part_data.empty:
                logging.warning(f"Part '{part}' not found, using default values")
                part_info = {
                    "part": part,
                    "part_type": "Unknown",
                    "part_price_inr": 5000.0,
                    "paint_cost_inr": 1000.0,
                    "fitment_hours": 1.0,
                    "paint_hours": 0.5,
                    "total_cost_inr": 6000.0
                }
            else:
                part_type = part_data['Part Type'].iloc[0]
                part_price = float(part_data[price_col].iloc[0])
                paint_cost = float(part_data[paint_col].iloc[0])
                fitment_hrs = float(part_data[fitment_col].iloc[0])
                paint_hrs = float(part_data[paint_hrs_col].iloc[0])
                part_total = part_price + paint_cost
                part_info = {
                    "part": part,
                    "part_type": part_type,
                    "part_price_inr": part_price,
                    "paint_cost_inr": paint_cost,
                    "fitment_hours": fitment_hrs,
                    "paint_hours": paint_hrs,
                    "total_cost_inr": part_total
                }
                total_cost += part_total
            parts_details.append(part_info)

        composite = generate_composite(original, mask)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        composite_hex = buffer.tobytes().hex()

        try:
            conn = sqlite3.connect(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\users.db")
            c = conn.cursor()
            result = {
                "description": description,
                "vehicle_type": vehicle_type,
                "parts_detected": [p["part"] for p in parts_details],
                "repair_cost": float(total_cost),
                "damage_severity": severity,
                "auto_approval": auto_approval,
                "estimate": estimate
            }
            c.execute("INSERT INTO history (username, image_path, result, timestamp) VALUES (?, ?, ?, ?)",
                      (username, f"Uploads/{file.filename}", json.dumps(result), 
                       datetime.now().isoformat()))
            conn.commit()
        except Exception as e:
            logging.error(f"Database error for {file.filename}: {e}")
        finally:
            conn.close()

        logging.info(f"Claim processed for {file.filename}: parts={detected_parts}, total_cost={total_cost}, severity={severity}, auto_approval={auto_approval}, estimate={estimate}")
        return {
            "result": result,
            "composite_image": composite_hex
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Claim processing failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Claim processing failed: {str(e)}")


@app.post("/repair-shops")
async def get_repair_shops(request: RepairShopRequest):
    try:
        city = request.city.lower().strip()
        insurance = request.insurance
        try:
            with open(r"C:\Users\paarth.sahni_infobea\Downloads\Car_new\repair_shops.json", "r") as f:
                shops = json.load(f)
        except FileNotFoundError:
            shops = [
                {"name": "AutoFix", "city": "mumbai", "insurance": ["ICICI", "HDFC"]},
                {"name": "CarCare", "city": "delhi", "insurance": ["Bajaj"]},
                {"name": "QuickRepair", "city": "bangalore", "insurance": ["SBI", "HDFC"]},
                {"name": "SpeedyFix", "city": "chennai", "insurance": ["ICICI"]}
            ]
            logging.warning("repair_shops.json not found, using default shops")
        
        if insurance:
            shops = [shop for shop in shops if shop["city"].lower() == city and insurance in shop["insurance"]]
        else:
            shops = [shop for shop in shops if shop["city"].lower() == city]
        
        logging.info(f"Returning shops for city: {city}, insurance: {insurance}")
        return {"city": city, "shops": shops}
    except Exception as e:
        logging.error(f"Failed to fetch repair shops: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch repair shops: {str(e)}")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logging.info("Running Uvicorn in async context")
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
            server = uvicorn.Server(config)
            loop.create_task(server.serve())
        else:
            logging.info("Starting FastAPI server with Uvicorn")
            uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise