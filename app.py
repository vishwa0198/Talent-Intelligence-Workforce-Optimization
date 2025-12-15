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
# STREAMLIT UI SETUP
# ===============================================================
st.set_page_config(layout="wide", page_title="T-IQ Dashboard")
st.title("T-IQ — Talent Intelligence & Workforce Optimization")

st.sidebar.header("Modules")
module = st.sidebar.radio(
    "Choose Module",
    [
        "HR Analytics",
        "Resume → Job Match",
        "Sentiment Analysis",
        "Document OCR & Classification",
        "HR Chatbot & LLM Tools",
        "View Models / Files",
    ]
)

# ===============================================================
# 1️⃣ HR ANALYTICS
# ===============================================================
if module == "HR Analytics":
    st.header("Attrition & Performance Prediction")
    uploaded = st.file_uploader("Upload Employee CSV Row", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Input:", df)

        try:
            perf_feat = models["performance"]["features"]

            # extract attrition feature names
            if hasattr(models["attrition"], "feature_names_in_"):   
                attr_feat = list(models["attrition"].feature_names_in_)
            else:
                attr_feat = perf_feat  # fallback

            # build union
            all_features = list(sorted(set(perf_feat + attr_feat)))

            # prepare X correctly
            X = prepare_employee_features(df, all_features)

        except Exception as e:
            st.error(f"Feature prep error: {e}")
            st.stop()

        # Performance Prediction
        if models["performance"]:
            try:
                pm = models["performance"]
                X_scaled = pm["scaler"].transform(X[pm["features"]])
                pred = pm["model"].predict(X_scaled)[0]
                st.metric("Predicted Performance", f"{pred:.2f}")
            except Exception as e:
                st.error(f"Performance Error: {e}")

        # Attrition Prediction
        if models["attrition"]:
            try:
                needed = list(models["attrition"].feature_names_in_)
                prob = models["attrition"].predict_proba(X[needed])[0, 1]
                st.metric("Attrition Risk", f"{prob:.3f}")
            except Exception as e:
                st.error(f"Attrition Error: {e}")

# ===============================================================
# 2️⃣ RESUME → JOB MATCH
# ===============================================================
elif module == "Resume → Job Match":
    st.header("Resume to Job Matching")

    resume_file = st.file_uploader("Upload Resume (txt/pdf)", type=["txt", "pdf"])
    top_n = st.slider("Top N Jobs", 1, 10, 5)

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

        st.write("Resume Preview:", text[:1000])

        try:
            jobs = pd.read_csv(os.path.join(PROCESSED, "job_descriptions_clean.csv"))
            
            # Debug: Show available columns
            st.write("Available columns in job data:", list(jobs.columns))
            
            # Check for common job title column names
            title_column = None
            possible_title_columns = ['positionName', 'job_title', 'title', 'position', 'role', 'Job Title']
            
            for col in possible_title_columns:
                if col in jobs.columns:
                    title_column = col
                    break
            
            if title_column is None:
                # If no standard title column found, use the first column or create a generic one
                if len(jobs.columns) > 0:
                    title_column = jobs.columns[0]
                    st.warning(f"No standard job title column found. Using '{title_column}' as job title.")
                else:
                    st.error("No columns found in job data.")
                    st.stop()
            
            # Check for description column
            desc_column = None
            possible_desc_columns = ['clean_description', 'description', 'job_description', 'desc']
            
            for col in possible_desc_columns:
                if col in jobs.columns:
                    desc_column = col
                    break
            
            if desc_column is None:
                st.error("No description column found in job data.")
                st.stop()
            
            job_texts = jobs[desc_column].astype(str).tolist()

            if models["embedder"] is None:
                st.error("Embedding model not loaded.")
                st.stop()

            emb_resume = models["embedder"].encode(text, convert_to_tensor=True)
            emb_jobs = models["embedder"].encode(job_texts, convert_to_tensor=True)

            sim = util.cos_sim(emb_resume, emb_jobs)[0]
            topk = torch.topk(sim, k=min(top_n, len(job_texts)))

            results = []
            for score, idx in zip(topk.values, topk.indices):
                idx = int(idx)   # Convert tensor to integer
                results.append({
                    "Job Title": jobs.iloc[idx][title_column],
                    "Score": float(score),
                    "Description": str(jobs.iloc[idx][desc_column])[:300] + "...",
                })

            st.table(pd.DataFrame(results))
            
        except FileNotFoundError:
            st.error(f"Job descriptions file not found at: {os.path.join(PROCESSED, 'job_descriptions_clean.csv')}")
        except Exception as e:
            st.error(f"Error processing job matching: {str(e)}")


# ===============================================================
# 3️⃣ SENTIMENT ANALYSIS
# ===============================================================
elif module == "Sentiment Analysis":
    st.header("Employee Review Sentiment")

    review = st.text_area("Paste employee review")
    if st.button("Predict Sentiment"):
        clf = models["sentiment"]["classifier"]
        embedder = models["embedder"]
        emb = embedder.encode([review])
        pred = clf.predict(emb)[0]

        st.success("Positive 👍" if pred == 1 else "Negative 👎")


# ===============================================================
# 4️⃣ DOCUMENT OCR & CLASSIFICATION
# ===============================================================
elif module == "Document OCR & Classification":
    st.header("Upload Document Image")

    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=500)

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        text = pytesseract.image_to_string(img)
        st.text(text[:500])

        doc_state = models["doc_state"]
        if doc_state:
            try:
                import torchvision.models as tv
                from torchvision import transforms

                # Debug: Show available keys in doc_state
                st.write("Available keys in model state:", list(doc_state.keys()))

                # Default document classes (common HR documents)
                default_classes = {
                    0: "Resume/CV",
                    1: "Job Description",
                    2: "Employment Contract",
                    3: "Performance Review",
                    4: "Leave Application",
                    5: "Offer Letter",
                    6: "ID Document",
                    7: "Other"
                }

                # Check for idx2label or similar mapping
                label_map = None
                ncls = None

                if "idx2label" in doc_state:
                    label_map = doc_state["idx2label"]
                    ncls = len(label_map)
                elif "class_to_idx" in doc_state:
                    # Reverse the mapping if class_to_idx exists
                    label_map = {v: k for k, v in doc_state["class_to_idx"].items()}
                    ncls = len(label_map)
                elif "classes" in doc_state:
                    label_map = {i: cls for i, cls in enumerate(doc_state["classes"])}
                    ncls = len(label_map)
                elif "num_classes" in doc_state:
                    ncls = doc_state["num_classes"]
                    label_map = {i: default_classes.get(i, f"Class_{i}") for i in range(ncls)}
                else:
                    # Try to infer number of classes from state_dict
                    if "state_dict" in doc_state:
                        state_dict = doc_state["state_dict"]
                    else:
                        state_dict = doc_state
                    
                    # Look for the final layer to determine number of classes
                    for key in state_dict.keys():
                        if "fc.weight" in key or "classifier.weight" in key or "fc.bias" in key:
                            if "weight" in key:
                                ncls = state_dict[key].shape[0]
                            elif "bias" in key:
                                ncls = state_dict[key].shape[0]
                            break
                    
                    if ncls is not None:
                        # Use default classes or generic labels
                        label_map = {i: default_classes.get(i, f"Document_Type_{i}") for i in range(ncls)}
                        st.info(f"Inferred {ncls} document classes from model architecture")
                    else:
                        # Last resort: assume common number of classes
                        ncls = 8
                        label_map = default_classes
                        st.warning(f"Could not determine classes from model. Using default {ncls} document types.")
                
                if label_map is None or ncls is None:
                    st.error("Could not determine document classes from model.")
                    st.stop()

                st.write(f"**Document Classes ({ncls} types):**")
                for idx, label in label_map.items():
                    st.write(f"  - {idx}: {label}")

                # Load model
                model = tv.resnet50(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, ncls)
                
                if "state_dict" in doc_state:
                    model.load_state_dict(doc_state["state_dict"])
                else:
                    model.load_state_dict(doc_state)
                
                model.eval()

                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                x = transform(img).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    logits = model(x)
                    pred = logits.argmax().item()
                    probs = torch.nn.functional.softmax(logits, dim=1)[0]
                
                st.success(f"**Predicted Document Type:** {label_map[pred]}")
                st.metric("Confidence", f"{probs[pred].item():.2%}")
                
                # Show all confidence scores
                st.write("**All Confidence Scores:**")
                prob_df = pd.DataFrame({
                    "Document Type": [label_map[i] for i in range(ncls)],
                    "Confidence": [f"{probs[i].item():.2%}" for i in range(ncls)],
                    "Score": [probs[i].item() for i in range(ncls)]
                })
                prob_df = prob_df.sort_values("Score", ascending=False)
                st.dataframe(prob_df[["Document Type", "Confidence"]], hide_index=True)
                        
            except Exception as e:
                st.error(f"Error during document classification: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("Document classifier not found.")


# ===============================================================
# 5️⃣ HR CHATBOT & LLM TOOLS
# ===============================================================
elif module == "HR Chatbot & LLM Tools":
    st.header("AI HR Assistant")

    tool = st.selectbox("Choose Tool", [
        "HR Chatbot",
        "Interview Transcript Summarizer",
        "Resume Improve+ment Feedback",
        "Job Description Generator",
        "Employee Review Analyzer",
    ])

    # Chatbot
    if tool == "HR Chatbot":
        msg = st.text_area("Ask HR Assistant")
        if st.button("Ask"):
            reply = chat_with_hr_bot(msg)
            st.write(reply["response"])

    # Interview Summarizer
    elif tool == "Interview Transcript Summarizer":
        file = st.file_uploader("Upload transcript", type=["txt", "pdf"])
        if file:
            tmp = "tmp_tr.pdf" if file.type == "application/pdf" else "tmp_tr.txt"
            with open(tmp, "wb") as f:
                f.write(file.read())
            txt = safe_read_pdf_bytes(tmp) if tmp.endswith(".pdf") else safe_read_txt(tmp)
            st.text_area("Summary", summarize_interview_transcript(txt), height=300)

    # Resume Feedback
    elif tool == "Resume Improvement Feedback":
        file = st.file_uploader("Upload resume", type=["txt", "pdf"])
        role = st.text_input("Target Role")
        if file:
            tmp = "tmp_r.pdf" if file.type == "application/pdf" else "tmp_r.txt"
            with open(tmp, "wb") as f:
                f.write(file.read())
            txt = safe_read_pdf_bytes(tmp) if tmp.endswith(".pdf") else safe_read_txt(tmp)
            st.text_area("Feedback", resume_improvement_feedback(txt, role_target=role))

    # JD Generator
    elif tool == "Job Description Generator":
        role = st.text_input("Role")
        skills = st.text_area("Skills")
        seniority = st.selectbox("Seniority", ["Junior", "Mid", "Senior"])
        if st.button("Generate"):
            st.text_area("Job Description", generate_job_description(role, skills, seniority))

    # Review Analyzer
    elif tool == "Employee Review Analyzer":
        review = st.text_area("Paste review")
        if st.button("Analyze"):
            st.text_area("Analysis", analyze_employee_review(review))


# ===============================================================
# 6️⃣ VIEW FILES / MODELS
# ===============================================================
elif module == "View Models / Files":
    st.header("Available Models & Files")
    st.write(os.listdir(MODELS_DIR))
    st.write(os.listdir(PROCESSED))
    st.write(os.listdir(OUTPUT_DIR))
