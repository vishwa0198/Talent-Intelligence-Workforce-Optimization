import streamlit as st
import pandas as pd
import joblib
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pytesseract
import os

# ================================
# IMPORT CHATBOT MODULE FUNCTIONS
# ================================
from chatbot import (
    chat_with_hr_bot,
    summarize_interview_transcript,
    resume_improvement_feedback,
    generate_job_description,
    analyze_employee_review,
    safe_read_pdf_bytes,
    safe_read_txt,
)

# ===============================================================
# DIRECTORIES
# ===============================================================
MODELS_DIR = "models"
PROCESSED = r"C:\Guvi\Talent Intelligence & Workforce Optimization\notebooks\processed"
OUTPUT_DIR = r"C:\Guvi\Talent Intelligence & Workforce Optimization\notebooks\output"

# ===============================================================
# MODEL LOADING
# ===============================================================
@st.cache_resource
def load_models():
    models = {}

    # Attrition
    try:
        file = [f for f in os.listdir(MODELS_DIR) if "attrition" in f][0]
        models["attrition"] = joblib.load(os.path.join(MODELS_DIR, file))
    except:
        models["attrition"] = None

    # Performance
    try:
        file = [f for f in os.listdir(MODELS_DIR) if "performance" in f][0]
        models["performance"] = joblib.load(os.path.join(MODELS_DIR, file))
    except:
        models["performance"] = None

    # Sentiment
    try:
        models["sentiment"] = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    except:
        models["sentiment"] = None

    # Document classifier
    try:
        file = [f for f in os.listdir(MODELS_DIR) if "document_classifier" in f][0]
        models["doc_state"] = torch.load(os.path.join(MODELS_DIR, file), map_location="cpu")
    except:
        models["doc_state"] = None

    # Embedding model
    try:
        models["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    except:
        models["embedder"] = None

    return models

models = load_models()


# ===============================================================
# UNIFIED FEATURE PREPARATION FUNCTION
# ===============================================================

def prepare_employee_features(df_raw: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Rebuild all required engineered columns for Attrition & Performance models.
    """
    df = df_raw.copy()

    X = pd.DataFrame(0, index=df.index, columns=feature_list)

    for idx, row in df.iterrows():

        # Copy numeric fields when present
        numeric_cols = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
            'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
            'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
            'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager', 'PerformanceRating'
        ]

        for col in numeric_cols:
            if col in feature_list:
                # SPECIAL FIX: Attrition model requires PerformanceRating
                if col == "PerformanceRating":
                    X.at[idx, col] = row.get(col, 3)   # default rating
                else:
                    X.at[idx, col] = row.get(col, 0)


        # Derived: TenureMonths
        if "TenureMonths" in feature_list:
            tenure = row.get("Tenure", row.get("YearsAtCompany", 0))
            X.at[idx, "TenureMonths"] = float(tenure) * 12.0

        # Gender_Male
        if "Gender_Male" in feature_list:
            gender = str(row.get("Gender", "")).lower()
            X.at[idx, "Gender_Male"] = 1 if gender in ["m", "male"] else 0

        # OverTime_Yes
        if "OverTime_Yes" in feature_list:
            overtime = str(row.get("OverTime", "")).lower()
            X.at[idx, "OverTime_Yes"] = 1 if overtime == "yes" else 0

        # One-hot category mapping
        mappings = {
            "BusinessTravel": "BusinessTravel_",
            "Department": "Department_",
            "EducationField": "EducationField_",
            "JobRole": "JobRole_",
            "MaritalStatus": "MaritalStatus_",
        }

        for raw_col, prefix in mappings.items():
            if raw_col in df.columns:
                value = str(row[raw_col]).strip().lower()
                for feat in feature_list:
                    if feat.startswith(prefix):
                        cat = feat.replace(prefix, "").strip().lower()
                        X.at[idx, feat] = 1 if cat == value else 0

    return X


# ===============================================================
# STREAMLIT UI SETUP & CUSTOM CSS
# ===============================================================
st.set_page_config(layout="wide", page_title="T-IQ | Talent Intelligence Suite", page_icon="🧠")

# Custom CSS for Professional UI
st.markdown("""
<style>
    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Card-like Containers */
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    h1 {
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Success/Error/Info Messages */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 5])
with col1:
    # Placeholder for Logo
    st.markdown("## 🧠") 
with col2:
    st.title("T-IQ — Talent Intelligence Suite")
    st.markdown("**AI-Powered Workforce Optimization & Analytics Dashboard**")

st.markdown("---")

# Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a module below:")

module = st.sidebar.radio(
    "",
    [
        "📊 HR Analytics",
        "📄 Resume → Job Match",
        "😊 Sentiment Analysis",
        "🔍 Document OCR",
        "🤖 AI Assistant",
        "📂 System Files"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online** ✅")

# ===============================================================
# 1️⃣ HR ANALYTICS
# ===============================================================
if "HR Analytics" in module:
    st.markdown("### 📊 Attrition & Performance Analytics")
    st.markdown("Predict employee flight risk and future performance scores using historical data.")
    
    with st.expander("ℹ️ How to use"):
        st.write("Upload a CSV file containing employee details (e.g., Age, Department, Salary, etc.) to get real-time predictions.")

    uploaded = st.file_uploader("Upload Employee Data (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown("#### Input Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Feature Prep & Prediction
        try:
            perf_feat = models["performance"]["features"]
            # extract attrition feature names
            if hasattr(models["attrition"], "feature_names_in_"):   
                attr_feat = list(models["attrition"].feature_names_in_)
            else:
                attr_feat = perf_feat  # fallback

            all_features = list(sorted(set(perf_feat + attr_feat)))
            X = prepare_employee_features(df, all_features)

        except Exception as e:
            st.error(f"❌ Feature preparation failed: {e}")
            st.stop()

        st.markdown("---")
        st.markdown("#### 🎯 Prediction Results")
        
        c1, c2 = st.columns(2)

        # Performance Prediction
        with c1:
            if models["performance"]:
                try:
                    pm = models["performance"]
                    X_scaled = pm["scaler"].transform(X[pm["features"]])
                    pred = pm["model"].predict(X_scaled)[0]
                    
                    st.metric("Predicted Performance Score", f"{pred:.2f}", delta="Target: 3.0+")
                    if pred < 3:
                        st.warning("⚠️ Low performance predicted.")
                    else:
                        st.success("✅ Good performance expected.")
                except Exception as e:
                    st.error(f"Performance Error: {e}")
            else:
                st.warning("Performance model not loaded.")

        # Attrition Prediction
        with c2:
            if models["attrition"]:
                try:
                    needed = list(models["attrition"].feature_names_in_)
                    prob = models["attrition"].predict_proba(X[needed])[0, 1]
                    
                    st.metric("Attrition Risk Probability", f"{prob:.1%}", delta_color="inverse")
                    
                    if prob > 0.5:
                        st.error("🚨 High Flight Risk Detected!")
                    else:
                        st.success("🛡️ Low Retention Risk")
                except Exception as e:
                    st.error(f"Attrition Error: {e}")
            else:
                st.warning("Attrition model not loaded.")

# ===============================================================
# 2️⃣ RESUME → JOB MATCH
# ===============================================================
elif "Resume → Job Match" in module:
    st.markdown("### 📄 Intelligent Resume Matching")
    st.markdown("Match candidate resumes against open job descriptions using semantic search.")

    c1, c2 = st.columns([1, 2])
    with c1:
        resume_file = st.file_uploader("Upload Profile (PDF/TXT)", type=["txt", "pdf"])
        top_n = st.slider("Number of Matches", 1, 10, 5)
    
    with c2:
        if resume_file:
            if resume_file.type == "text/plain":
                text = resume_file.read().decode()
            else:
                try:
                    import fitz
                    pdf = fitz.open(stream=resume_file.read(), filetype="pdf")
                    text = "".join([page.get_text() for page in pdf])
                except:
                    st.error("Cannot read PDF.")
                    text = ""
            st.text_area("Resume Preview", text, height=150)
        else:
            st.info("👈 Upload a resume to see matches here.")

    if resume_file:
        try:
            jobs = pd.read_csv(os.path.join(PROCESSED, "job_descriptions_clean.csv"))
            
            # Identify columns
            title_col = next((c for c in ['positionName', 'job_title', 'title', 'role'] if c in jobs.columns), jobs.columns[0])
            desc_col = next((c for c in ['clean_description', 'description', 'job_description'] if c in jobs.columns), None)

            if not desc_col:
                st.error("Job Database Error: No description column found.")
                st.stop()

            job_texts = jobs[desc_col].astype(str).tolist()

            if models["embedder"]:
                with st.spinner("Computing semantic similarity..."):
                    emb_resume = models["embedder"].encode(text, convert_to_tensor=True)
                    emb_jobs = models["embedder"].encode(job_texts, convert_to_tensor=True)

                    sim = util.cos_sim(emb_resume, emb_jobs)[0]
                    topk = torch.topk(sim, k=min(top_n, len(job_texts)))

                st.markdown("#### 🏆 Top Matched Roles")
                results = []
                for score, idx in zip(topk.values, topk.indices):
                    idx = int(idx)
                    results.append({
                        "Job Title": jobs.iloc[idx][title_col],
                        "Match Score": f"{float(score):.1%}",
                        "Preview": str(jobs.iloc[idx][desc_col])[:150] + "..."
                    })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)

            else:
                st.error("Embedding model unavailable.")
                
        except Exception as e:
            st.error(f"Error: {e}")

# ===============================================================
# 3️⃣ SENTIMENT ANALYSIS
# ===============================================================
elif "Sentiment Analysis" in module:
    st.markdown("### 😊 Employee Sentiment Analyzer")
    st.markdown("Analyze employee reviews, feedback, or survey responses using TextBlob NLP.")

    review = st.text_area("Enter textual feedback:", height=150, placeholder="Type review here...")
    
    if st.button("Analyze Sentiment"):
        try:
            from textblob import TextBlob
            blob = TextBlob(review)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            st.markdown("#### Analysis Result")
            
            # Determine sentiment category
            if polarity > 0.1:
                st.success(f"## Positive Sentiment 👍 (Score: {polarity:.2f})")
            elif polarity < -0.1:
                st.error(f"## Negative Sentiment 👎 (Score: {polarity:.2f})")
            else:
                st.warning(f"## Neutral Sentiment 😐 (Score: {polarity:.2f})")
                
            st.info(f"**Subjectivity:** {subjectivity:.2f} (0=Fact, 1=Opinion)")
            
        except ImportError:
            st.error("TextBlob not installed. Please run `pip install textblob`.")
        except Exception as e:
            st.error(f"Error: {e}")

# ===============================================================
# 4️⃣ DOCUMENT OCR
# ===============================================================
elif "Document OCR" in module:
    st.markdown("### 🔍 Document AI (OCR & Classification)")
    st.markdown("Automatically extract text and classify HR documents (Resumes, IDs, Contracts).")

    uploaded = st.file_uploader("Upload Document (Image)", type=["png", "jpg", "jpeg"])
    
    if uploaded:
        c1, c2 = st.columns(2)
        img = Image.open(uploaded).convert("RGB")
        
        with c1:
            st.image(img, caption="Uploaded Document", use_column_width=True)
        
        with c2:
            with st.spinner("Running OCR..."):
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                text = pytesseract.image_to_string(img)
            
            st.markdown("**Extracted Text:**")
            st.code(text[:1000])

                # Classification Logic
            doc_state = models["doc_state"]
            if doc_state:
                try:
                    import torchvision.models as tv
                    from torchvision import transforms

                    # --- Class Mapping Logic ---
                    # 1. Determine State Dict vs Full Model
                    if isinstance(doc_state, dict) and "state_dict" in doc_state:
                        state_dict = doc_state["state_dict"]
                    elif isinstance(doc_state, dict):
                        state_dict = doc_state
                    else:
                        st.error("Unknown model format.")
                        st.stop()

                    # 2. Infer Number of Classes (ncls) from weights
                    ncls = 8 # Fallback
                    if "fc.weight" in state_dict:
                        ncls = state_dict["fc.weight"].shape[0]
                    elif "classifier.weight" in state_dict: # MobileNet/VGG
                        ncls = state_dict["classifier.weight"].shape[0]
                    elif "fc.bias" in state_dict:
                        ncls = state_dict["fc.bias"].shape[0]
                    
                    st.info(f"Detected {ncls} document classes from model file.")

                    # 3. Define Label Map based on ncls
                    # If ncls is 4, we assume a specific subset (e.g., Resume, ID, etc.)
                    # Ideally, the model should store its own classes, but we can map common ones.
                    if ncls == 4:
                        default_classes = {0: "Resume", 1: "ID Card", 2: "Contract", 3: "Other"} # Example guess
                    else:
                        default_classes = {i: f"Type_{i}" for i in range(ncls)}
                        # Try to use standard mapping if ncls matches 8
                        if ncls == 8:
                            default_classes = {0: "Resume", 1: "Job Desc", 2: "Contract", 3: "Review", 4: "Leave App", 5: "Offer Letter", 6: "ID Card", 7: "Other"}

                    # 4. Load Model
                    model = tv.resnet50(weights=None)
                    model.fc = torch.nn.Linear(model.fc.in_features, ncls)
                    model.load_state_dict(state_dict)
                    model.eval()

                    # Transform
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)), transforms.CenterCrop(224),
                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    x = transform(img).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.nn.functional.softmax(logits, dim=1)[0]
                        pred_idx = logits.argmax().item()

                    st.markdown("#### Classification Result")
                    # Safe get
                    label_str = default_classes.get(pred_idx, f"Class {pred_idx}")
                    st.info(f"**Document Type:** {label_str}")
                    
                    st.progress(float(probs[pred_idx]))
                    st.caption(f"Confidence: {probs[pred_idx]:.1%}")
                    
                    with st.expander("See all scores"):
                         scores = {default_classes.get(i, f"C{i}"): float(probs[i]) for i in range(ncls)}
                         st.write(scores)

                except Exception as e:
                    st.warning(f"Classification skipped: {e}")

# ===============================================================
# 5️⃣ AI ASSISTANT
# ===============================================================
elif "AI Assistant" in module:
    st.markdown("### 🤖 Intelligent HR Assistant")
    st.markdown("Leverage GenAI for creating JDs, summarizing interviews, and answering HR queries.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chatbot", 
        "📝 Transcript Summary", 
        "📄 Resume Fixer", 
        "✍️ JD Generator",
        "📢 Review Analyzer"
    ])

    with tab1:
        st.markdown("#### HR Knowledge Bot")
        msg = st.text_area("Ask a question:", height=100)
        if st.button("Send", key="chat_btn"):
            with st.spinner("Thinking..."):
                reply = chat_with_hr_bot(msg)
            st.markdown(f"**Assistant:** {reply['response']}")

    with tab2:
        st.markdown("#### Interview Summarizer")
        tr_file = st.file_uploader("Upload Transcript", type=["txt", "pdf"], key="tr_up")
        if tr_file:
             # Process file wrapper
             tmp = "tmp_tr.pdf" if tr_file.type == "application/pdf" else "tmp_tr.txt"
             with open(tmp, "wb") as f: f.write(tr_file.read())
             txt = safe_read_pdf_bytes(tmp) if tmp.endswith(".pdf") else safe_read_txt(tmp)
             if st.button("Summarize"):
                 st.write(summarize_interview_transcript(txt))

    with tab3:
        st.markdown("#### Resume Improvement")
        res_file = st.file_uploader("Upload Resume", type=["txt", "pdf"], key="res_up")
        tgt_role = st.text_input("Target Role Name")
        if res_file and st.button("Analyze Resume"):
             tmp = "tmp_r.pdf" if res_file.type == "application/pdf" else "tmp_r.txt"
             with open(tmp, "wb") as f: f.write(res_file.read())
             txt = safe_read_pdf_bytes(tmp) if tmp.endswith(".pdf") else safe_read_txt(tmp)
             st.write(resume_improvement_feedback(txt, role_target=tgt_role))

    with tab4:
        st.markdown("#### Job Description Generator")
        c1, c2 = st.columns(2)
        r_name = c1.text_input("Role Title")
        r_lvl = c2.selectbox("Seniority", ["Entry", "Mid-Level", "Senior", "Executive"])
        r_skills = st.text_area("Key Skills / Tech Stack")
        if st.button("Generate JD"):
            st.markdown(generate_job_description(r_name, r_skills, r_lvl))

    with tab5:
        st.markdown("#### Performance Review Analyzer")
        rev_text = st.text_area("Paste Review Text")
        if st.button("Analyze Review"):
            st.write(analyze_employee_review(rev_text))

# ===============================================================
# 6️⃣ SYSTEM FILES
# ===============================================================
elif "System Files" in module:
    st.markdown("### 📂 System Diagnostics & File Explorer")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Models Directory**")
        st.code("\n".join(os.listdir(MODELS_DIR)))
    with c2:
        st.markdown("**Processed Data**")
        st.code("\n".join(os.listdir(PROCESSED)))
    with c3:
        st.markdown("**Output Artifacts**")
        st.code("\n".join(os.listdir(OUTPUT_DIR)))

