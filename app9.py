import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import logging
from requests.exceptions import ConnectionError, Timeout, RequestException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import base64
import torch
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import jwt
import uuid
import PyPDF2
import re
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

# Debug environment variables
logging.basicConfig(
    filename="streamlit_log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Streamlit app started")
logging.info(f"Env path: {env_path}")
logging.info(f"GEOAPIFY_API_KEY: {'Set' if os.getenv('GEOAPIFY_API_KEY') else 'Not set'}")
logging.info(f"GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}")
logging.info(f"GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
logging.info(f"SECRET_KEY: {'Set' if os.getenv('SECRET_KEY') else 'Not set'}")

# Configure Gemini API
try:
    # Temporary hardcode for debugging (REMOVE AFTER TESTING)
    api_key = "AIzaSyCmNmKCU9Fqas16_duc1aucJofzBGiIdyQ"
    # api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No API key found for Gemini. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    logging.info(f"Gemini 1.5 Flash model loaded successfully with API key: {'Hardcoded' if api_key == 'AIzaSyCmNmKCU9Fqas16_duc1aucJofzBGiIdyQ' else 'GOOGLE_API_KEY or GEMINI_API_KEY'}")
except Exception as e:
    logging.error(f"Failed to load Gemini model: {str(e)}")
    st.error(f"Failed to load Gemini model: {str(e)}. Ensure GOOGLE_API_KEY or GEMINI_API_KEY is set in .env.")
    raise


# Load SentenceTransformer model
try:
    with torch.no_grad():
        torch.set_default_device(torch.device("cpu"))
        model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device("cpu"))
        torch.set_default_device(None)
    logging.info("SentenceTransformer model loaded successfully on CPU")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer: {str(e)}")
    st.error(f"Failed to load embedding model: {str(e)}. Ensure 'sentence-transformers' and 'torch' are installed correctly.")
    st.stop()

# FastAPI endpoint and JWT secret
FASTAPI_URL = "http://localhost:8000"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-12345")

# Geoapify API Key
GEOAPIFY_API_KEY= "7e8b16100146459ebd9b0a5ee9bb3a32"
NOMINATIM_USER_AGENT = "auto_insurance_repair_finder_app_v5"

# Session state initialization - This block is CRUCIAL and CORRECTLY PLACED at the top in app7.py
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "claim_context" not in st.session_state:
    st.session_state.claim_context = {}
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "resolved_lat" not in st.session_state:
    st.session_state.resolved_lat = None
    st.session_state.resolved_lon = None
    st.session_state.resolved_full_address = ""
    st.session_state.search_performed = False
    st.session_state.last_search_radius = 5000
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}
if "claim_id" not in st.session_state:
    st.session_state.claim_id = None
if "policy_details" not in st.session_state:
    st.session_state.policy_details = {}
if "analysis_displayed" not in st.session_state:
    st.session_state.analysis_displayed = False
if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {"description": "", "vehicle_type": "hatchback"}
if "uploaded_chat_files" not in st.session_state:
    st.session_state.uploaded_chat_files = {}
# NEW: Session state for managing the chat file uploader's key to force reset
if "chat_uploader_key" not in st.session_state:
    st.session_state.chat_uploader_key = 0


# Modern CSS styling
st.markdown("""
    <style>
    .main {
        /* Main background: very light, clean base, subtly fading to muted light teal */
        background: linear-gradient(135deg, #F8FBFB, #88BDBC); /* Start with a near-white for freshness */
        color: #112D32; /* Very dark teal for general text on light background */
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        /* Buttons: deep teal background, light teal text, very dark teal border */
        background-color: #254E58;
        color: #88BDBC;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid #112D32;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(17, 45, 50, 0.2);
    }
    .stButton>button:hover {
        /* Button hover: muted dark brown/olive background for contrast, light teal text */
        background-color: #4F4A41;
        color: #88BDBC;
        box-shadow: 0 4px 8px rgba(17, 45, 50, 0.3);
    }
    .stFileUploader {
        /* File uploader: dashed border with muted light teal, transparent light background */
        border: 2px dashed #88BDBC;
        border-radius: 10px;
        padding: 20px;
        background-color: rgba(248, 251, 251, 0.8); /* Semi-transparent near-white */
        color: #112D32;
    }
    h1 {
        /* Heading 1: very dark teal */
        color: #112D32;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 0.5em;
        text-shadow: 0 1px 2px rgba(17, 45, 50, 0.1);
    }
    h2, h3 {
        /* Headings 2, 3: deep teal */
        color: #254E58;
        font-weight: 600;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        /* Text inputs: transparent light background, muted light teal border, very dark teal text */
        border-radius: 8px;
        border: 1px solid #88BDBC;
        padding: 10px;
        background-color: #F8FBFB; /* Near-white background */
        color: #112D32;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>input:focus, .stTextArea>div>textarea:focus {
        /* Text input focus: deep teal border, subtle shadow */
        border-color: #254E58;
        box-shadow: 0 0 0 2px rgba(37, 78, 88, 0.3);
    }
    .stSelectbox {
        /* Selectbox: transparent light background, muted light teal border, very dark teal text */
        border-radius: 8px;
        border: 1px solid #88BDBC;
        background-color: #F8FBFB;
        color: #112D32;
    }
    .stDataFrame, .result-card {
        /* Dataframes and result cards: muted light teal border, semi-transparent light background */
        border: 1px solid #88BDBC;
        border-radius: 10px;
        background-color: rgba(248, 251, 251, 0.85); /* Slightly more opaque near-white */
        padding: 20px;
        margin-bottom: 10px;
        color: #112D32; /* Very dark teal for text in cards */
        box-shadow: 0 2px 4px rgba(17, 45, 50, 0.1);
        max-width: 100%;
        width: auto;
        box-sizing: border-box;
        overflow-x: hidden;
        overflow-wrap: break-word;
        word-break: break-word;
    }
    .result-card h4 {
        margin: 0 0 10px 0;
        color: #254E58; /* Deep teal for card headings */
        font-weight: 600;
    }
    .result-card p {
        margin: 5px 0;
        color: #112D32; /* Very dark teal for text in cards */
        overflow-wrap: break-word;
    }
    .result-card ul {
        margin: 10px 0 10px 30px;
        padding: 0;
        list-style-type: disc;
        overflow-wrap: break-word;
    }
    .result-card li {
        margin-bottom: 5px;
        padding-right: 15px;
        overflow-wrap: break-word;
        word-break: break-word;
    }
    .sidebar .sidebar-content {
        /* Sidebar: deep teal gradient for a distinct, sophisticated navigation area */
        background: linear-gradient(180deg, #112D32, #254E58);
        color: #88BDBC; /* Light text on dark sidebar */
    }
    .sidebar .stButton>button {
        /* Sidebar buttons: deep teal background, light teal text, light border */
        background-color: #254E58;
        color: #88BDBC;
        width: 100%;
        border: 1px solid #88BDBC;
    }
    .sidebar .stButton>button:hover {
        /* Sidebar button hover: muted dark brown/olive background, light teal text */
        background-color: #4F4A41;
        color: #88BDBC;
        box-shadow: 0 4px 8px rgba(17, 45, 50, 0.3);
    }
    .stChatMessage {
        /* Chat messages: semi-transparent muted light teal, very dark teal text */
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        background-color: rgba(136, 189, 188, 0.85); /* Slightly more opaque #88BDBC */
        color: #112D32; /* Dark text for readability on light background */
        box-shadow: 0 2px 4px rgba(17, 45, 50, 0.1);
    }
    /* Specific styling for user messages in chat for better differentiation */
    .stChatMessage[data-test-id="stChatMessage-user"] {
        background-color: rgba(79, 74, 65, 0.1); /* Very subtle transparent warm neutral for user */
        color: #112D32; /* Dark text */
        margin-left: auto; /* Push user message to the right */
        border-bottom-right-radius: 2px; /* Slight design variation */
        border-left: 3px solid #254E58; /* Accent border */
    }
    .stChatMessage[data-test-id="stChatMessage-model"] {
        background-color: rgba(136, 189, 188, 0.85); /* Light teal for model messages */
        color: #112D32; /* Dark text */
        border-bottom-left-radius: 2px; /* Slight design variation */
        border-right: 3px solid #4F4A41; /* Accent border for model messages */
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def classify_severity(description, parts_detected=None):
    try:
        prompt = f"""
        Assess the damage severity for a vehicle based on the following:
        - Description: {description}
        - Parts Detected: {', '.join(parts_detected) if parts_detected else 'None'}
        Return one of: 'light', 'moderate', 'severe'. Provide a brief explanation.
        """
        response = gemini_model.generate_content(prompt)
        severity_text = response.text.strip().lower()
        severity_match = re.search(r'(light|moderate|severe)', severity_text)
        severity = severity_match.group(1) if severity_match else "moderate"
        logging.info(f"Gemini severity assessment: {severity} for description: {description}")
        return severity
    except Exception as e:
        logging.error(f"Gemini severity assessment failed: {str(e)}. Falling back to rule-based.")
        if any(keyword in description.lower() for keyword in ["scratch", "minor dent"]):
            return "light"
        elif any(keyword in description.lower() for keyword in ["bumper off", "crushed"]):
            return "severe"
        return "moderate"

def extract_policy_details(pdf_file):
    try:
        # User's provided implementation using PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
        logging.info(f"Extracted PDF text: {text[:500]}...") # Log first 500 chars
        
        # Enhanced regex for policy limit
        limit_match = re.search(r'\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|\bDollars\b)|\b\d+\s*(?:K|Thousand|Million)\b', text, re.IGNORECASE)
        if limit_match:
            limit_text = limit_match.group(0).replace(",", "").lower()
            if "million" in limit_text:
                policy_limit = f"${int(float(limit_text.replace('million', '').strip())) * 1000000}"
            elif "thousand" in limit_text or "k" in limit_text:
                policy_limit = f"${int(float(limit_text.replace('thousand', '').replace('k', '').strip())) * 1000}"
            else:
                policy_limit = limit_text.replace("usd", "").replace("dollars", "").strip()
        else:
            # Fallback: any number resembling a dollar amount
            fallback_match = re.search(r'\b\d{4,}\b', text)
            policy_limit = f"${fallback_match.group(0)}" if fallback_match else "$0"
        
        # Enhanced regex for policyholder
        username_match = re.search(r'(?:Policyholder|Insured|Name):\s*([\w\s\'-]+)', text, re.IGNORECASE)
        policyholder = username_match.group(1).strip() if username_match else ""
        
        logging.info(f"Policy parsed: policy_limit={policy_limit}, policyholder={policyholder}")
        return {"policy_limit": policy_limit, "policyholder": policyholder}
    except Exception as e:
        logging.error(f"Error extracting policy details: {str(e)}")
        st.error(f"Failed to parse policy document: {str(e)}")
        return {"policy_limit": "$0", "policyholder": ""}


# Helper function for PDF text extraction (using PyPDF2 as used elsewhere)
def extract_text_from_pdf(pdf_bytes):
    """Extracts text from PDF bytes using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

@st.cache_data
def load_geo_data():
    try:
        with open("data/geo.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("geo.json not found, using default data")
        return {
            "India": {
                "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
                "Delhi": ["Delhi"],
                "Karnataka": ["Bangalore"],
                "Tamil Nadu": ["Chennai"]
            }
        }

def get_location_from_ip():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        city = res.get("city", "")
        region = res.get("region", "")
        return region, city
    except:
        return "", ""

def get_lat_lon_from_query(query_string):
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    try:
        location = geolocator.geocode(query_string, language="en", timeout=10)
        if location:
            time.sleep(1.1) # Respect Nominatim usage policy
            return location.latitude, location.longitude, location.address
        else:
            time.sleep(1.1)
            return None, None, None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        time.sleep(1.1)
        return None, None, None
    except GeocoderServiceError as e:
        st.error(f"Geocoding service error: {e}. This might be a temporary issue or rate limit.")
        time.sleep(1.1)
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during geocoding: {e}")
        time.sleep(1.1)
        return None, None, None

def find_nearby_repair_shops_geoapify(latitude, longitude, radius_meters=5000):
    if not GEOAPIFY_API_KEY:
        st.error("Geoapify API key is missing. Please ensure GEOAPIFY_API_KEY is set in .env file.")
        logging.error("Geoapify API key missing in environment variables")
        return []
    shops = []
    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": "service.vehicle.repair.car",
        "filter": f"circle:{longitude},{latitude},{radius_meters}",
        "limit": 20,
        "apiKey": GEOAPIFY_API_KEY,
        "lang": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and data.get("features"):
            for feature in data["features"]:
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                name = properties.get("name", "Unnamed Repair Shop")
                address = properties.get("formatted", "Address not specified")
                lon = geometry["coordinates"][0] if geometry and "coordinates" in geometry else None
                lat = geometry["coordinates"][1] if geometry and "coordinates" in geometry else None
                shops.append({
                    "name": name,
                    "address": address,
                    "lat": lat,
                    "lon": lon,
                    "phone": properties.get("datasource", {}).get("raw", {}).get("phone"),
                    "website": properties.get("website")
                })
        return shops
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Geoapify API: {e}")
        logging.error(f"Geoapify API request failed: {str(e)}")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding JSON from Geoapify API response.")
        logging.error("Geoapify API returned invalid JSON")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred with Geoapify API: {e}")
        logging.error(f"Geoapify API unexpected error: {str(e)}")
        return []

# RAG Knowledge Base
knowledge_base = [
    {
        "id": "policy_coverage_limit",
        "type": "policy_clause",
        "content": "Your comprehensive policy has a maximum payout limit of $100,000 for a single claim. This limit applies to all covered damages, including repairs and associated costs. Remember, your deductible will be applied first.",
        "keywords": ["policy limit", "payout", "maximum", "comprehensive", "limit"]
    },
    {
        "id": "policy_deductible",
        "type": "policy_clause",
        "content": "A standard deductible of $5,000 applies to all claims. This is the amount you pay out-of-pocket before your insurance coverage begins. This helps manage premiums and shared responsibility.",
        "keywords": ["deductible", "out-of-pocket", "claim cost", "excess payment"]
    },
    {
        "id": "repair_minor_dent",
        "type": "repair_cost_guide",
        "content": "Minor dents (less than 6 inches in diameter, without paint damage to underlying metal) typically cost between $500 and $1,000 to repair, depending on the vehicle model, paint type, and location of the damage. For specific estimates, we'd need detailed photos or a physical inspection.",
        "keywords": ["minor dent", "cost", "repair", "small damage", "dent cost"]
    },
    {
        "id": "repair_bumper_replacement",
        "type": "repair_cost_guide",
        "content": "Full bumper replacement can range from $1,500 to $3,500, significantly varying based on the vehicle's make, model, type of bumper (e.g., standard, reinforced), and integration of sensors or cameras. This estimate doesn't include painting costs which are extra.",
        "keywords": ["bumper", "replacement", "cost", "severe damage", "bumper cost"]
    },
    {
        "id": "documents_required",
        "type": "procedure",
        "content": "For all claims, a police report (especially if the incident involved another vehicle, injury, or significant public property damage) and at least two detailed repair estimates from certified shops are required for a final assessment and claim processing. These documents are crucial for determining liability and the true extent of damage.",
        "keywords": ["documents", "police report", "repair estimates", "required", "proof", "report", "bill"]
    },
    {
        "id": "settlement_process",
        "type": "procedure",
        "content": "Once all required documents (police report, repair estimates) are submitted and reviewed, our team will make a final assessment. We will then communicate our settlement offer, which will be based on your policy terms and the validated repair costs. You'll have an opportunity to review and accept the offer.",
        "keywords": ["settlement", "process", "claim process", "final offer"]
    }
]

def retrieve_info(query, kb, top_n=1):
    query_lower = query.lower()
    relevant_docs = []
    for doc in kb:
        if any(keyword in query_lower for keyword in doc["keywords"]):
            relevant_docs.append(doc["content"])
            break
    return "\n".join(relevant_docs) if relevant_docs else ""

# Login Function
def login():
    st.title("🛡️ AI-Powered Insurance Management System")
    st.markdown("**Your intelligent partner for seamless claims and vehicle management.**")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_username", placeholder="Enter username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
        if st.button("Login", key="login_button"):
            if not username or not password:
                st.error("Username and password are required")
                logging.error("Login attempt with empty username or password")
                return
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/login",
                    data={"username": username, "password": password},
                    timeout=10
                )
                if response.status_code == 200:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.session_id = str(uuid.uuid4())
                    token = jwt.encode(
                        {"username": username, "session_id": st.session_state.session_id},
                        SECRET_KEY,
                        algorithm="HS256"
                    )
                    st.session_state.jwt_token = token
                    st.success("Logged in successfully!")
                    logging.info(f"User {username} logged in, session_id: {st.session_state.session_id}")
                    st.rerun()
                else:
                    error_msg = response.json().get("detail", "Invalid credentials")
                    st.error(error_msg)
                    logging.error(f"Login failed for {username}: {error_msg}")
            except ConnectionError:
                st.error("Cannot connect to FastAPI server. Ensure it is running at http://localhost:8000")
                logging.error("ConnectionError: FastAPI server not reachable")
            except Timeout:
                st.error("Request to FastAPI server timed out")
                logging.error("Timeout: FastAPI server request timed out")
            except RequestException as e:
                st.error(f"Failed to connect to backend: {str(e)}")
                logging.error(f"RequestException during login: {str(e)}")
    
    with tab2:
        st.subheader("Create a New Account")
        reg_username = st.text_input("New Username", key="reg_username", placeholder="Choose a username")
        reg_password = st.text_input("New Password", type="password", key="reg_password", placeholder="Choose a password")
        if st.button("Register", key="register_button"):
            if not reg_username or not reg_password:
                st.error("Username and password are required")
                logging.error("Register attempt with empty username or password")
                return
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/register",
                    data={"username": reg_username, "password": reg_password},
                    timeout=10
                )
                if response.status_code == 200:
                    st.success("Registered successfully! Please log in.")
                    logging.info(f"User {reg_username} registered successfully")
                else:
                    error_msg = response.json().get("detail", "Registration failed")
                    st.error(error_msg)
                    logging.error(f"Registration failed for {reg_username}: {error_msg}")
            except ConnectionError:
                st.error("Cannot connect to FastAPI server. Ensure it is running at http://localhost:8000")
                logging.error("ConnectionError: FastAPI server not reachable")
            except Timeout:
                st.error("Request to FastAPI server timed out")
                logging.error("Timeout: FastAPI server request timed out")
            except RequestException as e:
                st.error(f"Failed to connect to backend: {str(e)}")
                logging.error(f"RequestException during registration: {str(e)}")

# Main App
def main_app():
    st.title("🛡️ AI-Powered Insurance Management System")
    st.markdown("**Your intelligent partner for seamless claims and vehicle management.**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload & Analyze", "🛠️ Repair Shops", "💬 Chatbot", "✅ Auto-Approval"])
    
    with tab1:
        st.subheader("Upload Vehicle Image")
        description = st.text_input(
            "Damage Description",
            value=st.session_state.form_inputs["description"],
            placeholder="e.g., Front headlight damaged",
            key="description"
        )
        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["hatchback", "sedan", "suv", "muv", "premium"],
            index=["hatchback", "sedan", "suv", "muv", "premium"].index(st.session_state.form_inputs["vehicle_type"]),
            key="vehicle_type"
        )
        st.session_state.form_inputs.update({"description": description, "vehicle_type": vehicle_type})
        
        uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_uploader")
        
        if st.session_state.analysis_data and st.session_state.analysis_displayed:
            analysis_data = st.session_state.analysis_data
            st.subheader("Analysis Results")
            st.markdown("""
                <div class="result-card">
                    <h4>Damage Analysis</h4>
                    <p><strong>Description:</strong> {}</p>
                    <p><strong>Vehicle Type:</strong> {}</p>
                    <p><strong>Damage Type:</strong> {}</p>
                    <p><strong>Parts Detected:</strong></p>
                    <ul>
            """.format(
                analysis_data["description"],
                analysis_data["vehicle_type"],
                analysis_data["damage_type"]
            ), unsafe_allow_html=True)
            for part, price in analysis_data["part_prices"].items():
                st.markdown(f"<li>{part}: ₹{price}</li>", unsafe_allow_html=True)
            st.markdown("""
                    </ul>
                    <p><strong>Total Repair Cost:</strong> ₹{}</p>
                </div>
            """.format(analysis_data["total_cost"]), unsafe_allow_html=True)
            if "composite_image" in analysis_data:
                st.image(
                    Image.open(io.BytesIO(analysis_data["composite_image"])),
                    caption="Damage Analysis (Red: Damage, Green: Non-Damage)",
                    width=300
                )
            elif "image_bytes" in analysis_data:
                st.image(
                    Image.open(io.BytesIO(analysis_data["image_bytes"])),
                    caption="Uploaded Image (No Analysis Available)",
                    width=300
                )
        
        if uploaded_file:
            try:
                file_bytes = uploaded_file.getvalue()
                if not file_bytes:
                    st.error("Uploaded file is empty")
                    logging.error(f"Empty file uploaded: {uploaded_file.name}")
                    return
                image = Image.open(io.BytesIO(file_bytes))
                image.verify()
                image = Image.open(io.BytesIO(file_bytes)) # Re-open after verify as verify closes the file
                logging.info(f"Image validated: {uploaded_file.name}, type: {uploaded_file.type}, size: {len(file_bytes)} bytes")
            except Exception as e:
                st.error(f"Invalid image file: {str(e)}")
                logging.error(f"Invalid image in Streamlit: {uploaded_file.name}, error: {str(e)}")
                return
            
            if st.button("Analyze Damage", key="analyze_button"):
                if not description:
                    st.error("Damage description is required")
                    logging.error("Analyze attempt with empty description")
                    return
                if not st.session_state.username:
                    st.error("User not logged in")
                    logging.error("Analyze attempt without logged-in user")
                    return
                with st.spinner("Analyzing damage..."):
                    mime_type = uploaded_file.type
                    if mime_type == "image/jpg":
                        mime_type = "image/jpeg"
                    files = {"file": (uploaded_file.name, file_bytes, mime_type)}
                    data = {
                        "description": description,
                        "vehicle_type": vehicle_type,
                        "username": st.session_state.username
                    }
                    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
                    try:
                        response = requests.post(
                            f"{FASTAPI_URL}/process_claim",
                            files=files,
                            data=data,
                            headers=headers,
                            timeout=15
                        )
                        if response.status_code == 200:
                            data = response.json()
                            result = data["result"]
                            parts_detected = result.get("parts_detected", [])
                            repair_cost = result.get("repair_cost", 0)
                            if isinstance(repair_cost, str):
                                repair_cost = repair_cost.replace("₹", "").strip()
                                total_cost = int(float(repair_cost))
                            elif isinstance(repair_cost, (int, float)):
                                total_cost = int(repair_cost)
                            else:
                                total_cost = 0
                            part_prices = {part: total_cost // len(parts_detected) if parts_detected else 0 for part in parts_detected}
                            
                            severity = result.get("severity") or result.get("damage_type")
                            if not severity:
                                severity = classify_severity(description, parts_detected)
                            
                            st.session_state.analysis_data = {
                                "description": description,
                                "vehicle_type": vehicle_type,
                                "damage_type": severity,
                                "image_bytes": file_bytes,
                                "part_prices": part_prices,
                                "total_cost": total_cost,
                                "parts_detected": parts_detected,
                                "composite_image": bytes.fromhex(data["composite_image"]) if "composite_image" in data else None
                            }
                            st.session_state.analysis_displayed = True
                            
                            st.subheader("Analysis Results")
                            st.markdown("""
                                <div class="result-card">
                                    <h4>Damage Analysis</h4>
                                    <p><strong>Description:</strong> {}</p>
                                    <p><strong>Vehicle Type:</strong> {}</p>
                                    <p><strong>Damage Type:</strong> {}</p>
                                    <p><strong>Parts Detected:</strong></p>
                                    <ul>
                            """.format(description, vehicle_type, severity), unsafe_allow_html=True)
                            for part, price in part_prices.items():
                                st.markdown(f"<li>{part}: ₹{price}</li>", unsafe_allow_html=True)
                            st.markdown("""
                                    </ul>
                                    <p><strong>Total Repair Cost:</strong> ₹{}</p>
                                </div>
                            """.format(total_cost), unsafe_allow_html=True)
                            
                            if "composite_image" in st.session_state.analysis_data and st.session_state.analysis_data["composite_image"]:
                                composite_image_to_display = Image.open(io.BytesIO(st.session_state.analysis_data["composite_image"]))
                                st.image(composite_image_to_display, caption="Damage Analysis (Red: Damage, Green: Non-Damage)", width=300)
                            logging.info(f"Image analysis successful for {uploaded_file.name}, total_cost: ₹{total_cost}, severity: {severity}")
                        else:
                            error_msg = response.json().get("detail", "Error analyzing image")
                            st.error(f"Error analyzing image: {error_msg}")
                            logging.error(f"Image analysis failed for {uploaded_file.name}: {error_msg}, status_code={response.status_code}")
                    except ConnectionError:
                        st.error("Cannot connect to FastAPI server. Ensure it is running at http://localhost:8000")
                        logging.error("ConnectionError: FastAPI server not reachable during analysis")
                    except Timeout:
                        st.error("Request to FastAPI server timed out")
                        logging.error("Timeout: Request to FastAPI server timed out during analysis")
                    except RequestException as e:
                        st.error(f"Failed to connect to backend: {str(e)}")
                        logging.error(f"RequestException during analysis: {str(e)}")
        
        st.subheader("Analysis History")
        if st.button("View History", key="history_button"):
            headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
            try:
                response = requests.get(
                    f"{FASTAPI_URL}/history",
                    params={"username": st.session_state.username},
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    history = response.json()
                    if history:
                        df = pd.DataFrame([
                            {
                                "Timestamp": item["timestamp"],
                                "Description": json.loads(item["result"])["description"],
                                "Vehicle Type": json.loads(item["result"])["vehicle_type"],
                                "Parts Detected": ", ".join(json.loads(item["result"])["parts_detected"]),
                                "Repair Cost": f"₹{int(float(json.loads(item['result']).get('repair_cost', 0)))}"
                            }
                            for item in history
                        ])
                        st.dataframe(df, use_container_width=True)
                        
                        st.subheader("Download Report")
                        report_buffer = io.BytesIO()
                        c = canvas.Canvas(report_buffer, pagesize=letter)
                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(100, 750, "Car Damage Detection Report")
                        c.setFont("Helvetica", 12)
                        c.drawString(100, 730, f"User: {st.session_state.username}")
                        y = 700
                        for item in history:
                            result = json.loads(item["result"])
                            c.drawString(100, y, f"Timestamp: {item['timestamp']}")
                            c.drawString(100, y-20, f"Description: {result['description'][:50]}...")
                            c.drawString(100, y-40, f"Vehicle Type: {result['vehicle_type']}")
                            c.drawString(100, y-60, f"Parts: {', '.join(result['parts_detected'])}")
                            c.drawString(100, y-80, f"Repair Cost: ₹{int(float(result.get('repair_cost', 0)))}")
                            y -= 100
                            if y < 100:
                                c.showPage()
                                y = 750
                        c.save()
                        report_buffer.seek(0)
                        st.download_button(
                            label="Download PDF Report",
                            data=report_buffer,
                            file_name="damage_report.pdf",
                            mime="application/pdf",
                            key="download_report"
                        )
                    else:
                        st.info("No history found")
                        logging.info(f"No history found for user {st.session_state.username}")
                else:
                    error_msg = response.json().get("detail", "Error fetching history")
                    st.error(error_msg)
                    logging.error(f"History fetch failed: {error_msg}")
            except ConnectionError:
                st.error("Cannot connect to FastAPI server. Ensure it is running at http://localhost:8000")
                logging.error("ConnectionError: FastAPI server not reachable during history fetch")
            except Timeout:
                st.error("Request to FastAPI server timed out")
                logging.error("Timeout: Request to FastAPI server timed out during history fetch")
            except RequestException as e:
                st.error(f"Failed to connect to backend: {str(e)}")
                logging.error(f"RequestException during history fetch: {str(e)}")

    with tab2:
        st.subheader("Find Repair Shops")
        geo_data = load_geo_data()
        countries = list(geo_data.keys())
        selected_country = "India"
        selected_state_display = ""
        selected_city_display = ""
        final_location_query = ""

        mode = st.radio("Choose Location Mode", ["📍 Autodetect", "✋ Manual"], horizontal=True, key="repair_mode")

        if mode == "📍 Autodetect":
            st.info("Detecting your location automatically...")
            detected_state, detected_city = get_location_from_ip()
            st.text_input("Country", selected_country, disabled=True, key="auto_country")
            st.text_input("Detected State", detected_state, disabled=True, key="auto_state")
            st.text_input("Detected City", detected_city, disabled=True, key="auto_city")
            if detected_state and selected_country in geo_data and detected_state in geo_data[selected_country]:
                selected_state_display = detected_state
                if detected_city in geo_data[selected_country][selected_state_display]:
                    selected_city_display = detected_city
                    st.success(f"Detected city '{detected_city}' found in our data for {detected_state}.")
                else:
                    selected_city_display = geo_data[selected_country][selected_state_display][0] if geo_data[selected_country][selected_state_display] else ""
                    if selected_city_display:
                        st.warning(f"Your exact detected city '{detected_city}' wasn't found in our list for {detected_state}. Using '{selected_city_display}' as a fallback.")
            else:
                st.warning("Could not precisely autodetect your state/city in our data. Please use manual selection or provide more details.")
                selected_state_display = "Maharashtra"
                selected_city_display = "Mumbai"
                st.info(f"Defaulting to: {selected_city_display}, {selected_state_display}")
            user_address_refinement = st.text_input(
                "Enter a more specific address, landmark, or locality (e.g., 'Vijay Nagar', 'Near railway station')",
                value="", key="refine_address"
            )
            if user_address_refinement:
                base_city = detected_city if detected_city else selected_city_display
                base_state = detected_state if detected_state else selected_state_display
                final_location_query = f"{user_address_refinement}, {base_city}, {base_state}, {selected_country}"
                st.info(f"Using refined query for search: **{final_location_query}**")
            else:
                final_location_query = f"{selected_city_display}, {selected_state_display}, {selected_country}"
                st.info(f"Using detected/default location for search: **{selected_city_display}, {selected_state_display}**")
        else:
            selected_country = st.selectbox("Select Country", countries, index=countries.index("India"), key="manual_country")
            if selected_country and selected_country in geo_data:
                states = list(geo_data[selected_country])
                states.sort()
                selected_state_display = st.selectbox("Select State", states, key="manual_state")
                if selected_state_display and selected_state_display in geo_data[selected_country]:
                    cities = geo_data[selected_country][selected_state_display]
                    cities.sort()
                    selected_city_display = st.selectbox("Select City", cities, key="manual_city")
                else:
                    st.warning("No cities found for the selected state.")
                    selected_city_display = ""
            else:
                st.warning("No states found for the selected country.")
                selected_state_display = ""
            if selected_city_display and selected_state_display:
                final_location_query = f"{selected_city_display}, {selected_state_display}, {selected_country}"
            else:
                final_location_query = ""

        if st.button("🔍 Search Repair Shops", key="shops_button"):
            if final_location_query:
                st.session_state.resolved_lat, st.session_state.resolved_lon, st.session_state.resolved_full_address = get_lat_lon_from_query(final_location_query)
                st.session_state.search_performed = True
                if st.session_state.resolved_lat and st.session_state.resolved_lon:
                    st.success(f"📍 Location resolved to: **{st.session_state.resolved_full_address if st.session_state.resolved_full_address else 'Unknown Address'}** → Latitude: {st.session_state.resolved_lat}, Longitude: {st.session_state.resolved_lon}")
                else:
                    st.error("❌ Unable to fetch coordinates. Please check your inputs or try a different location.")
                    st.session_state.search_performed = False
            else:
                st.warning("Please select/enter a location before searching.")
                st.session_state.search_performed = False

        if st.session_state.search_performed and st.session_state.resolved_lat is not None and st.session_state.resolved_lon is not None:
            st.subheader("🛠️ Nearby Repair Shops")
            search_radius = st.slider(
                "Search Radius (meters)", 
                500, 
                20000, 
                st.session_state.get('last_search_radius', 5000),
                step=500,
                key="radius_slider"
            )
            st.session_state.last_search_radius = search_radius
            shops = find_nearby_repair_shops_geoapify(
                st.session_state.resolved_lat, 
                st.session_state.resolved_lon, 
                radius_meters=search_radius
            )
            if shops:
                st.success(f"Found {len(shops)} repair shops!")
                shops_df = pd.DataFrame(shops)
                shops_df = shops_df.dropna(subset=['lat', 'lon'])
                if not shops_df.empty:
                    center_df = pd.DataFrame([{
                        'lat': st.session_state.resolved_lat,
                        'lon': st.session_state.resolved_lon,
                        'name': 'Your Location'
                    }])
                    all_points_df = pd.concat([center_df, shops_df[['lat', 'lon', 'name']]], ignore_index=True)
                    st.map(all_points_df, zoom=12, use_container_width=True)
                else:
                    st.warning("No shops with valid coordinates to display on the map.")
                
                st.markdown("### Nearest Shops Are")
                for idx, shop in enumerate(shops, 1):
                    st.markdown(f"{idx}. **{shop['name']}**")
                    st.markdown(f"    - Address: {shop['address']}")
                    if shop['phone']: st.markdown(f"    - Phone: {shop['phone']}")
                    if shop['website']: st.markdown(f"    - Website: {shop['website']}")
                logging.info(f"Found {len(shops)} shops for {final_location_query}, radius={search_radius}")
            else:
                st.info(f"No repair shops found within {search_radius} meters of your location. Try broadening your search radius.")
                logging.info(f"No shops found for {final_location_query}, radius={search_radius}")

    with tab3: # Ensure this 'with tab3:' is correctly indented within your main_app() function
        st.subheader("Negotiate Claim")

        # --- Initial check for analysis data (prerequisite) ---
        if not st.session_state.get("analysis_data"):
            st.warning("Please analyze a vehicle in the 'Upload & Analyze' tab first.")
            st.stop() 

        analysis_data = st.session_state.analysis_data

        # --- Claim ID Display ---
        if not st.session_state.get("claim_id"):
            st.session_state.claim_id = f"CLM-{str(uuid.uuid4())[:8]}"
        st.markdown(f"**Claim ID**: `{st.session_state.claim_id}`")

        # --- Claim Details Card ---
        st.markdown(f"""
            <div class="result-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h4 style="color: #333; margin-top: 0;">Claim Details</h4>
                <p><strong>Damage Severity:</strong> <span style="color: red;">{analysis_data.get("damage_type", "N/A")}</span></p>
                <p><strong>Total Repair Cost:</strong> ₹{analysis_data.get("total_cost", "N/A")}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Claim Image Display ---
        if analysis_data and "image_bytes" in analysis_data and analysis_data["image_bytes"]:
            try:
                st.image(
                    Image.open(io.BytesIO(analysis_data["image_bytes"])),
                    caption="Claim Image (from Analysis)",
                    width=200
                )
            except Exception as e:
                st.error(f"Could not display claim image: {e}")
                logging.error(f"Error displaying claim image from analysis: {e}")
        else:
            st.info("No claim image available from analysis.")

        # --- New Negotiation Button ---
        if st.button("🆕 Start New Negotiation", key="new_chat_button"):
            st.session_state.chat_history = []
            st.session_state.claim_context = {}
            st.session_state.chat_started = False
            st.session_state.policy_details = None
            st.session_state.claim_id = f"CLM-{str(uuid.uuid4())[:8]}"
            st.session_state.uploaded_chat_files = {} 
            # Reset the file uploader key to ensure it's empty for a new negotiation
            st.session_state.chat_uploader_key = 0 
            logging.info(f"User {st.session_state.get('username', 'Guest')} started new negotiation, claim_id: {st.session_state.claim_id}")
            st.rerun()

        # --- Policy Upload Section ---
        st.markdown("---")
        st.subheader("Upload Policy Document")
        policy_file = st.file_uploader("Upload Policy Document (PDF)", type=["pdf"], key="policy_uploader")
        if policy_file:
            try:
                policy_details = extract_policy_details(policy_file)
                st.session_state.policy_details = policy_details
                st.success(f"Policy loaded: Limit {policy_details.get('policy_limit', 'N/A')}")
                if policy_details.get("policyholder", "").lower() != st.session_state.get("username", "guest").lower():
                    st.warning(f"Policyholder name ({policy_details.get('policyholder', 'N/A')}) does not match current user ({st.session_state.get('username', 'guest')}). Proceeding anyway.")
                    logging.warning(f"Policyholder mismatch for user {st.session_state.get('username', 'guest')}: policyholder={policy_details.get('policyholder', 'N/A')}")
            except Exception as e:
                st.error(f"Error processing policy document: {e}")
                logging.error(f"Policy extraction error: {e}")
                st.session_state.policy_details = None
        elif not st.session_state.get("policy_details") and st.session_state.chat_started:
            st.warning("No policy document uploaded or processed yet for this negotiation.")

        # --- Chat Negotiation Section ---
        st.markdown("---")
        st.subheader("Chat Negotiation")

        chat_messages_display_container = st.container(height=400, border=True)

        if not st.session_state.chat_started:
            if st.button("Start Negotiation Chat", key="start_chat_button", disabled=not st.session_state.get("policy_details")):
                if not st.session_state.get("policy_details"):
                    st.error("Please upload and process a policy document before starting the chat.")
                    st.stop()

                try: # Outer try-except for general chat setup errors
                    st.session_state.claim_context = {
                        "claim_id": st.session_state.claim_id,
                        "damage_type": analysis_data.get("damage_type", "Unknown"),
                        "description": analysis_data.get("description", "No description provided."),
                        "vehicle_type": analysis_data.get("vehicle_type", "Unknown"),
                        "total_cost": analysis_data.get("total_cost", 0),
                        "parts_detected": analysis_data.get("parts_detected", []),
                        "policy_limit": st.session_state.policy_details.get("policy_limit", "$0")
                    }

                    initial_user_parts = [{"text": analysis_data.get("description", "Vehicle damage detected.")}]
                    file_bytes = analysis_data.get("image_bytes")
                    if file_bytes:
                        try: # Try-except for initial image processing
                            image = Image.open(io.BytesIO(file_bytes))
                            img_byte_arr = io.BytesIO()
                            img_format = "PNG"
                            image.save(img_byte_arr, format=img_format)
                            initial_user_parts.append({
                                "inline_data": {
                                    "mime_type": f"image/{img_format.lower()}",
                                    "data": base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                                }
                            })
                            logging.info("Image included in initial chat message")
                        except Exception as e:
                            st.warning(f"Error preparing image for initial chat: {str(e)}. Continuing without image.")
                            logging.error(f"Image processing error for initial chat message: {str(e)}")

                    st.session_state.chat_history.append({"role": "user", "parts": initial_user_parts})
                    
                    context = st.session_state.claim_context
                    policy_limit_usd = 0
                    try: # Try-except for policy limit parsing
                        policy_limit_str = context.get("policy_limit", "$0").replace("$", "").strip()
                        policy_limit_usd = int(float(policy_limit_str))
                    except ValueError:
                        st.warning(f"Could not parse policy limit: {context.get('policy_limit', 'N/A')}. Defaulting to $0.")
                        logging.warning(f"Policy limit parsing error: {context.get('policy_limit', 'N/A')}")

                    total_cost_inr = context.get("total_cost", 0)
                    total_cost_usd = round(total_cost_inr / 83) if total_cost_inr else 0
                    
                    # --- REVISED: Initial System Prompt for Gemini ---
                    initial_system_prompt = f"""
                    You are 'Your AI Claim advisor', an AI negotiation chatbot designed to assist the policyholder (the user) in getting the best possible settlement for their car damage claim.
                    Your goal is to analyze all provided information (damage details, policy, and uploaded documents like repair estimates or photos) to suggest a fair and optimal price for the user to propose to their insurance company.

                    You are capable of directly reviewing the content of uploaded documents. When documents (like repair estimates or police reports) are uploaded, I will provide their extracted text content (for PDFs/documents) or indicate their presence (for images). You must use this document content directly in your assessment and advice.

                    Claim Details (initial assessment from damage analyzer):
                    - Claim ID: {context['claim_id']}
                    - Damage Type: {context['damage_type']}
                    - Description: {context['description']}
                    - Vehicle Type: {context['vehicle_type']}
                    - Initial Estimated Total Repair Cost: ₹{total_cost_inr} (~${total_cost_usd})
                    - Parts Detected: {', '.join(context['parts_detected']) if context['parts_detected'] else 'None'}
                    - Policy Limit: {context['policy_limit']}

                    Your first response should acknowledge the user's claim and the initial damage assessment. Then, **explain your role as their helper to maximize their claim value within policy limits**, and guide them on what information you need (e.g., policy details if not already uploaded, detailed repair estimates if they don't have one, additional photos).

                    Crucially, when a repair estimate or new information is provided:
                    1.  **Extract relevant cost details** (e.g., total repair cost, itemized parts and labor) from the document content.
                    2.  **Compare this new cost to the policy limit.**
                    3.  **Provide immediate, actionable advice to the user:**
                        * State a preliminary assessment of whether the claim appears *covered and justifiable* based on the provided estimate and policy limit.
                        * **Suggest a negotiation price range** or a specific price for the user to propose to their insurance company, based on your analysis of the estimate and policy.
                        * Do not ask for documents that have just been uploaded.
                        * If further information is needed, specify it clearly.
                    """
                    
                    if gemini_model:
                        with st.spinner("Starting chat..."):
                            logging.info(f"DEBUG: Attempting Gemini generate_content with initial prompt. Prompt length: {len(initial_system_prompt)} chars.")
                            try: # Nested try-except specifically for the Gemini API call
                                response = gemini_model.generate_content(contents=[{"role": "user", "parts": [{"text": initial_system_prompt}]}])
                                logging.info(f"DEBUG: Gemini generate_content call completed successfully for initial chat.")
                                bot_reply = response.text.strip()
                                st.session_state.chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})
                                logging.info(f"Initial chat response generated for user {st.session_state.get('username', 'Guest')}: {bot_reply[:100]}...")
                            except Exception as gemini_api_e:
                                st.error(f"Error generating initial chat response from Gemini: {gemini_api_e}. Please check your API key, network, or prompt size.")
                                logging.error(f"ERROR: Gemini API call failed during initial chat: {gemini_api_e}", exc_info=True)
                                st.session_state.chat_started = False # Do not start chat if initial response fails
                                st.stop() # Stop further execution to make error visible
                    else:
                        st.error("Gemini model not initialized. Cannot start chat.")
                        logging.error(f"Gemini model is None when attempting to start chat for user {st.session_state.get('username', 'Guest')}.")
                        st.session_state.chat_started = False
                        st.stop()
                    
                    # --- Send claim data to backend ---
                    headers = {"Authorization": f"Bearer {st.session_state.get('jwt_token', '')}"}
                    files = {"file": ("image.png", file_bytes, "image/png")} if file_bytes else {}
                    data = {
                        "username": st.session_state.get("username", "GuestUser"),
                        "description": context.get('description', ''),
                        "vehicle_type": context.get("vehicle_type", ''),
                        "parts_detected": ",".join(context.get("parts_detected", [])) if context.get("parts_detected") else "",
                        "repair_cost": str(total_cost_inr)
                    }
                    try: # Try-except for the requests.post call
                        with st.spinner("Saving claim details..."):
                            response = requests.post(
                                f"{FASTAPI_URL}/process_claim",
                                files=files,
                                data=data,
                                headers=headers,
                                timeout=30
                            )
                        if response.status_code == 200:
                            st.success("Claim details saved successfully!")
                            logging.info(f"Chat claim saved for user {st.session_state.get('username', 'Guest')}: {st.session_state.claim_id}")
                        else:
                            error_msg = response.json().get("detail", f"Error saving claim (Status: {response.status_code})")
                            st.error(f"Failed to save claim: {error_msg}")
                            logging.error(f"Failed to save chat claim for user {st.session_state.get('username', 'Guest')}: {error_msg}")
                    except requests.exceptions.Timeout:
                        st.error("Timeout: Could not save claim details to the server. Please check your network or server status.")
                        logging.error(f"Timeout saving chat claim for user {st.session_state.get('username', 'Guest')}.")
                    except requests.exceptions.ConnectionError:
                        st.error("Connection Error: Could not reach the server to save claim details. Is FastAPI running?")
                        logging.error(f"Connection error saving chat claim for user {st.session_state.get('username', 'Guest')}.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred while saving claim: {str(e)}")
                        logging.error(f"Unexpected error saving chat claim for user {st.session_state.get('username', 'Guest')}: {str(e)}")

                    st.session_state.chat_started = True
                    st.rerun()

                except Exception as e: # Catch for general errors during initial chat setup
                    st.error(f"An error occurred while setting up the chat: {str(e)}")
                    logging.error(f"Setup chat error for user {st.session_state.get('username', 'Guest')}: {str(e)}", exc_info=True)
                    st.session_state.chat_started = False

        else: # Chat is started (this 'else' must be aligned with its 'if not st.session_state.chat_started:')
            with chat_messages_display_container:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        for part in msg["parts"]:
                            if "text" in part:
                                display_text = part['text']
                                if msg["role"] == "user" and "Additional Context" in display_text and "---" in display_text:
                                    user_message_start = display_text.find("User's message:")
                                    if user_message_start != -1:
                                        display_text = display_text[user_message_start + len("User's message:"):].strip()
                                st.markdown(display_text)
                            elif "inline_data" in part:
                                try:
                                    img_data = Image.open(io.BytesIO(base64.b64decode(part["inline_data"]["data"])))
                                    caption = f"Uploaded File ({part['inline_data'].get('mime_type', 'image')})"
                                    st.image(img_data, caption=caption, width=300)
                                except Exception as e:
                                    st.warning(f"Could not display image in chat history: {e}")
                                    logging.error(f"Error displaying inline image in chat history: {e}")
                st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

            st.markdown("---")
            st.write("### Attach Files")
            # Pass the chat_uploader_key to force re-rendering and clearing the uploader
            uploaded_chat_files_list = st.file_uploader(
                "Upload documents or additional photos for the adjuster:",
                type=["pdf", "jpg", "jpeg", "png", "doc", "docx", "xls", "xlsx"],
                accept_multiple_files=True,
                key=f"chat_file_uploader_{st.session_state.chat_uploader_key}" # Unique key
            )

            if uploaded_chat_files_list: # This block executes if new files are selected in the uploader
                new_files_for_chat_display = []
                text_from_documents = []

                for uploaded_file in uploaded_chat_files_list:
                    # Check if file has already been processed in this session to prevent infinite re-processing
                    if uploaded_file.name not in st.session_state.uploaded_chat_files:
                        try:
                            file_bytes = uploaded_file.getvalue()
                            mime_type = uploaded_file.type
                            
                            st.session_state.uploaded_chat_files[uploaded_file.name] = {
                                "bytes": file_bytes,
                                "mime_type": mime_type
                            }
                            
                            if mime_type.startswith("image/"):
                                new_files_for_chat_display.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": base64.b64encode(file_bytes).decode("utf-8")
                                    }
                                })
                                logging.info(f"Image '{uploaded_file.name}' attached via chat file uploader.")
                            elif mime_type == "application/pdf":
                                logging.info(f"Attempting to extract text from PDF: {uploaded_file.name}")
                                pdf_text = extract_text_from_pdf(file_bytes)
                                if pdf_text:
                                    text_from_documents.append(f"--- Content of {uploaded_file.name} ---\n{pdf_text}\n--- End of {uploaded_file.name} ---")
                                    new_files_for_chat_display.append({"text": f"Uploaded document: {uploaded_file.name} (PDF - Text Extracted)"})
                                    logging.info(f"PDF '{uploaded_file.name}' uploaded and text extracted ({len(pdf_text)} chars).")
                                else:
                                    new_files_for_chat_display.append({"text": f"Uploaded document: {uploaded_file.name} (PDF - Text Extraction Failed)"})
                                    logging.warning(f"Failed to extract text from PDF: {uploaded_file.name}")
                            else:
                                new_files_for_chat_display.append({"text": f"Uploaded document: {uploaded_file.name} ({uploaded_file.type}) - No text extraction implemented"})
                                logging.info(f"Document '{uploaded_file.name}' attached via chat file uploader (no text extraction).")

                        except Exception as e:
                            st.warning(f"Error processing uploaded file '{uploaded_file.name}': {e}. Skipping this file.")
                            logging.error(f"Error processing chat attached file: {e}", exc_info=True)
                
                if new_files_for_chat_display:
                    st.session_state.chat_history.append({"role": "user", "parts": [{"text": "I have uploaded the following documents/photos:"}] + new_files_for_chat_display})

                    gemini_user_upload_parts = []
                    if text_from_documents:
                        gemini_user_upload_parts.append({"text": "Please analyze the following document content:\n\n" + "\n\n".join(text_from_documents)})
                    
                    for part in new_files_for_chat_display:
                        if "inline_data" in part and part["inline_data"]["mime_type"].startswith("image/"):
                            gemini_user_upload_parts.append(part)
                    
                    if not gemini_user_upload_parts:
                        gemini_user_upload_parts.append({"text": "I have uploaded files. Please review."})

                    if gemini_model:
                        with st.spinner("Processing uploaded files..."):
                            temp_history_for_upload_response = list(st.session_state.chat_history) + [{"role": "user", "parts": gemini_user_upload_parts}]
                            logging.info(f"Attempting Gemini generate_content for file upload. History length: {len(temp_history_for_upload_response)}, current parts length: {len(gemini_user_upload_parts)}")
                            try: # Nested try-except specifically for this Gemini call
                                response = gemini_model.generate_content(contents=temp_history_for_upload_response)
                                logging.info(f"DEBUG: Gemini generate_content call completed successfully for file upload.")
                                bot_reply = response.text.strip()
                                st.session_state.chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})
                                logging.info(f"Bot responded to file upload for user {st.session_state.get('username', 'Guest')}: {bot_reply[:100]}...")
                            except Exception as gemini_api_e:
                                st.error(f"Error generating response from Gemini for uploaded files: {gemini_api_e}. Check API key, network, or file content.")
                                logging.error(f"ERROR: Gemini API call failed for file upload: {gemini_api_e}", exc_info=True)
                                # If Gemini fails, remove the last user message about file upload to avoid misleading history
                                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                    st.session_state.chat_history.pop()
                    else:
                        st.error("Gemini model not initialized. Cannot process uploaded files.")
                        logging.error(f"Gemini model is None when processing uploaded files for user {st.session_state.get('username', 'Guest')}.")
                
                # IMPORTANT: Increment the uploader key and rerun to clear the file uploader widget
                st.session_state.chat_uploader_key += 1
                logging.info(f"DEBUG: Incrementing chat_uploader_key to {st.session_state.chat_uploader_key} and rerunning after file upload.")
                st.rerun()

            user_msg = st.chat_input("Type your message...", key="chat_message_input")
            
            if user_msg:
                new_user_parts = [{"text": user_msg}]
                st.session_state.chat_history.append({"role": "user", "parts": new_user_parts})

                try:
                    retrieved_context = ""
                    if user_msg and knowledge_base:
                        retrieved_context = retrieve_info(user_msg, knowledge_base)
                    elif not knowledge_base:
                        logging.warning("Knowledge base not initialized. RAG will not be applied.")

                    rag_augmented_user_msg = user_msg
                    if retrieved_context:
                        rag_augmented_user_msg = f"""
                        Additional Context: {retrieved_context}
                        ---
                        User's message: {user_msg}
                        """
                    
                    gemini_input_parts_for_text_turn = []
                    if rag_augmented_user_msg:
                        gemini_input_parts_for_text_turn.append({"text": rag_augmented_user_msg})

                    temp_gemini_chat_history = []
                    for msg_h in st.session_state.chat_history:
                        current_turn_parts_for_gemini = []
                        for part in msg_h["parts"]:
                            if "text" in part:
                                if msg_h["role"] == "user" and "Additional Context" in part["text"] and "---" in part["text"]:
                                    current_turn_parts_for_gemini.append({"text": part["text"]})
                                else:
                                    current_turn_parts_for_gemini.append(part)
                            elif "inline_data" in part:
                                current_turn_parts_for_gemini.append(part)
                        temp_gemini_chat_history.append({"role": msg_h["role"], "parts": current_turn_parts_for_gemini})

                    if gemini_model:
                        with st.spinner("Generating response..."):
                            logging.info(f"Attempting Gemini generate_content for text input. History length: {len(temp_gemini_chat_history)}, current parts length: {len(gemini_input_parts_for_text_turn)}")
                            try: # Nested try-except for this Gemini call
                                response = gemini_model.generate_content(contents=temp_gemini_chat_history)
                                logging.info(f"DEBUG: Gemini generate_content call completed successfully for text input.")
                                bot_reply = response.text.strip()
                                st.session_state.chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})
                                logging.info(f"Chat response generated for user {st.session_state.get('username', 'Guest')}: {bot_reply[:100]}...")
                            except Exception as gemini_api_e:
                                st.error(f"Error generating response from Gemini for text input: {gemini_api_e}. Check API key, network, or prompt size.")
                                logging.error(f"ERROR: Gemini API call failed for text input: {gemini_api_e}", exc_info=True)
                                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                    st.session_state.chat_history.pop()

                except Exception as e:
                    st.error(f"Error processing chat message: {str(e)}")
                    logging.error(f"Chat error for user {st.session_state.get('username', 'Guest')}: {str(e)}", exc_info=True)
                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                        st.session_state.chat_history.pop()
                
                st.rerun()

            else:
                st.info("Upload a policy document and click 'Start Negotiation Chat' to begin your conversation with the Adjuster.")


    with tab4:
        st.subheader("Auto-Approval")
        if not st.session_state.analysis_data:
            st.warning("Please analyze a vehicle in the 'Upload & Analyze' tab first.")
            return
        
        analysis_data = st.session_state.analysis_data
        st.markdown("""
            <div class="result-card">
                <h4>Current Vehicle</h4>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Vehicle Type:</strong> {vehicle_type}</p>
                <p><strong>Damage Type:</strong> {damage_type}</p>
                <p><strong>Parts Detected:</strong> {parts_detected}</p>
                <p><strong>Total Repair Cost:</strong> ₹{total_cost}</p>
            </div>
        """.format(
            description=analysis_data.get("description", "N/A"),
            vehicle_type=analysis_data.get("vehicle_type", "N/A"),
            damage_type=analysis_data.get("damage_type", "N/A"),
            parts_detected=", ".join(analysis_data.get("parts_detected", [])),
            total_cost=analysis_data.get("total_cost", "N/A")
        ), unsafe_allow_html=True)

        if analysis_data.get("image_bytes"):
            st.image(
                Image.open(io.BytesIO(analysis_data["image_bytes"])),
                caption="Vehicle Image",
                width=200
            )
        
        if st.button("Check Auto-Approval Status", key="auto_submit_button"):
            severity = analysis_data.get("damage_type")
            total_cost_auto_approve = analysis_data.get("total_cost", 0)

            is_approved = False
            approval_reason = ""

            if severity == "light" and total_cost_auto_approve <= 50000:
                is_approved = True
                approval_reason = "Light damage and repair cost within auto-approval limits."
            elif severity == "moderate" and total_cost_auto_approve <= 100000:
                is_approved = True
                approval_reason = "Moderate damage and repair cost within auto-approval limits."
            else:
                approval_reason = "Damage severity or repair cost exceeds auto-approval criteria."

            status = "Approved for Auto-Approval" if is_approved else "Requires Manual Review"
            
            st.markdown(f"**Auto-Approval Status**: {status}")
            st.info(f"Reason: {approval_reason}")
            logging.info(f"Auto-approval result for user {st.session_state.get('username', 'Guest')}: {status} - {approval_reason}")
            
            headers = {"Authorization": f"Bearer {st.session_state.get('jwt_token', '')}"}
            files = {"file": ("image.png", analysis_data["image_bytes"], "image/png")} if analysis_data.get("image_bytes") else {}
            data = {
                "username": st.session_state.get("username", "GuestUser"),
                "description": analysis_data.get("description", ""),
                "vehicle_type": analysis_data.get("vehicle_type", ""),
                "parts_detected": ",".join(analysis_data.get("parts_detected", [])),
                "repair_cost": str(analysis_data.get("total_cost", 0)),
                "auto_approval_status": status,
                "auto_approval_reason": approval_reason
            }
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/process_claim",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=15
                )
                if response.status_code == 200:
                    st.success("Claim details (including auto-approval status) saved successfully!")
                    logging.info(f"Auto-approval claim saved for user {st.session_state.get('username', 'Guest')}")
                else:
                    error_msg = response.json().get("detail", "Error saving claim")
                    st.error(f"Error saving claim: {error_msg}")
                    logging.error(f"Failed to save auto-approval claim: {error_msg}")
            except Exception as e:
                st.error(f"Error during auto-approval save: {str(e)}")
                logging.error(f"Error during auto-approval save: {str(e)}")
        
        if st.button("Check Another Vehicle", key="another_vehicle_button"):
            st.session_state.analysis_data = {}
            st.session_state.analysis_displayed = False
            st.session_state.form_inputs = {"description": "", "vehicle_type": "hatchback"}
            st.info("Please return to the 'Upload & Analyze' tab to analyze a new vehicle.")
            logging.info(f"User {st.session_state.get('username', 'Guest')} requested to check another vehicle")

# App Logic
if not st.session_state.logged_in:
    login()
else:
    st.sidebar.markdown(f"### Welcome, {st.session_state.username}")
    st.sidebar.markdown(f"Session ID: {st.session_state.session_id}")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.session_id = None
        st.session_state.jwt_token = None
        st.session_state.analysis_data = {}
        st.session_state.chat_history = []
        st.session_state.claim_context = {}
        st.session_state.chat_started = False
        st.session_state.claim_id = None
        st.session_state.policy_details = {}
        st.session_state.analysis_displayed = False
        st.session_state.form_inputs = {"description": "", "vehicle_type": "hatchback"}
        logging.info(f"User {st.session_state.get('username', 'Guest')} logged out")
        st.rerun()
    main_app()