import streamlit as st
import numpy as np
import cv2
import json
import os
import psycopg2
from PIL import Image
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="FaceShield | Biometric System",
    page_icon="🛡️",
    layout="centered"
)

# --- Configuration ---
# Tries to get DB URL from Streamlit Secrets first, then Environment Variables
DATABASE_URL = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL"))
SIMILARITY_THRESHOLD = 0.5

# --- Caching Resources (Critical for Streamlit Performance) ---
@st.cache_resource
def load_models():
    """
    Loads ML models once and caches them in memory.
    This prevents reloading models on every user interaction.
    """
    print("Loading models...")
    # 1. MTCNN for detection
    mtcnn = MTCNN(keep_all=False, device='cpu')
    
    # 2. ArcFace for embeddings
    arcface = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    arcface.prepare(ctx_id=0, det_size=(640, 640))
    
    return mtcnn, arcface

@st.cache_resource
def init_db():
    """Initializes the database connection pool or table creation."""
    if not DATABASE_URL:
        st.warning("DATABASE_URL not found. Data will not be saved.")
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(255) PRIMARY KEY,
                embedding TEXT NOT NULL
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database Config Error: {e}")

# Load resources
mtcnn, arcface = load_models()
init_db()

# --- Helper Functions ---
def process_image(uploaded_file):
    """Converts uploaded file to format needed for MTCNN (PIL) and ArcFace (CV2)"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # For CV2 (ArcFace)
    img_cv = cv2.imdecode(file_bytes, 1)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # For PIL (MTCNN)
    img_pil = Image.open(uploaded_file).convert("RGB")
    
    # Reset file pointer for subsequent reads if necessary
    uploaded_file.seek(0)
    
    return img_pil, img_cv

def get_embedding(img_rgb):
    faces = arcface.get(img_rgb)
    if len(faces) == 0:
        return None
    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# --- UI Layout ---
st.title("🛡️ Face Shield")
st.markdown("Secure Biometric Verification System")

# Create Tabs
tab1, tab2 = st.tabs(["👤 New Registration", "🔐 Verify Identity"])

# --- TAB 1: REGISTRATION ---
with tab1:
    st.header("Register New User")
    
    with st.form("register_form"):
        reg_username = st.text_input("Username / ID", placeholder="e.g. john_doe")
        reg_file = st.file_uploader("Upload Reference Photo", type=['jpg', 'png', 'jpeg'], key="reg_img")
        
        submit_reg = st.form_submit_button("Save Identity")
        
        if submit_reg:
            if not reg_username or not reg_file:
                st.error("Please provide both username and image.")
            else:
                with st.spinner("Processing biometric data..."):
                    try:
                        pil_img, cv_img = process_image(reg_file)
                        
                        # 1. Check Face (MTCNN)
                        if mtcnn(pil_img) is None:
                            st.error("❌ No face detected in the image.")
                            st.stop()

                        # 2. Generate Embedding
                        embedding = get_embedding(cv_img)
                        if embedding is None:
                            st.error("❌ High-quality face extraction failed.")
                            st.stop()

                        # 3. Save to DB
                        conn = get_db_connection()
                        cur = conn.cursor()
                        
                        # Check exist
                        cur.execute("SELECT username FROM users WHERE username = %s", (reg_username,))
                        if cur.fetchone():
                            st.error(f"User '{reg_username}' already exists!")
                        else:
                            embedding_json = json.dumps(embedding.tolist())
                            cur.execute("INSERT INTO users (username, embedding) VALUES (%s, %s)", (reg_username, embedding_json))
                            conn.commit()
                            st.success(f"✅ User '{reg_username}' registered successfully!")
                            
                        cur.close()
                        conn.close()

                    except Exception as e:
                        st.error(f"Error: {e}")

# --- TAB 2: VERIFICATION ---
with tab2:
    st.header("Verify Identity")
    
    ver_username = st.text_input("Claimed ID", placeholder="e.g. john_doe")
    ver_file = st.file_uploader("Upload Live Photo", type=['jpg', 'png', 'jpeg'], key="ver_img")
    
    verify_btn = st.button("Verify Access")
    
    if verify_btn and ver_file and ver_username:
        # Display Image Side-by-Side (Optional visual aid)
        st.image(ver_file, caption="Live Upload", width=200)

        with st.spinner("Verifying..."):
            try:
                # 1. Fetch User Data
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT embedding FROM users WHERE username = %s", (ver_username,))
                result = cur.fetchone()
                cur.close()
                conn.close()

                if not result:
                    st.error("❌ User not found in database.")
                else:
                    stored_embedding = np.array(json.loads(result[0]))
                    
                    # 2. Process Uploaded Image
                    pil_img, cv_img = process_image(ver_file)
                    
                    if mtcnn(pil_img) is None:
                        st.error("❌ No face detected in uploaded photo.")
                    else:
                        new_embedding = get_embedding(cv_img)
                        if new_embedding is None:
                            st.error("❌ Could not extract facial features.")
                        else:
                            # 3. Compare
                            similarity = cosine_similarity([stored_embedding], [new_embedding])[0][0]
                            
                            # Display Result
                            col1, col2 = st.columns(2)
                            col1.metric("Similarity Score", f"{similarity:.4f}")
                            
                            if similarity >= SIMILARITY_THRESHOLD:
                                col2.success("✅ MATCH CONFIRMED")
                                st.balloons()
                            else:
                                col2.error("❌ NO MATCH")
                                
            except Exception as e:
                st.error(f"System Error: {e}")
