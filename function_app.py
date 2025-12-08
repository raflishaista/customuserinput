import azure.functions as func
import logging
import json
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="convertexceltojson", auth_level=func.AuthLevel.FUNCTION)
def convertexceltojson(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file_bytes = None

        # Try JSON first
        try:
            body = req.get_json()
            if "$content" in body:
                logging.info("Detected JSON input with $content")
                file_bytes = base64.b64decode(body["$content"])
            else:
                logging.warning("JSON body found but missing '$content'")
                return func.HttpResponse(
                    "Missing '$content' in request body",
                    status_code=400
                )
        except ValueError:
            # If JSON parsing fails, fall back to raw binary
            logging.info("Falling back to binary input")
            file_bytes = req.get_body()

        if not file_bytes:
            return func.HttpResponse("No file data found", status_code=400)

        # Debug log file size
        logging.info(f"Received file size: {len(file_bytes)} bytes")

        # Read Excel into DataFrame
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

        # Debug info
        logging.info(f"DataFrame shape: {df.shape}")

        result_json = df.to_json(orient="records")

        return func.HttpResponse(
            result_json,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error converting Excel to JSON: {e}")
        return func.HttpResponse(str(e), status_code=500)
    
# Global model cache
model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def minmax_scaling(series):
    if series.max() == series.min():
        return 0.5
    return (series - series.min()) / (series.max() - series.min())

def convert_to_months(duration):
    years, months = 0, 0
    parts = str(duration).split()
    for part in parts:
        if 'tahun' in part:
            try: years = int(parts[parts.index(part)-1])
            except (ValueError, IndexError): years = 0
        elif 'bulan' in part:
            try: months = int(parts[parts.index(part)-1])
            except (ValueError, IndexError): months = 0
    return years * 12 + months

def load_datasets_from_json(json_data):
    """Convert uploaded datasets JSON → dictionary of pandas DataFrames."""
    loaded = {}
    filename_mapping = {
        "(Pseudonym) Assignment Data.xlsx": "df_assign",
        "Usecase Requirement.xlsx": "df_ureq",
        "(Pseudonym) Talent Data.xlsx": "df_talent",
        "(Pseudonym) Skill Inventory.xlsx": "df_skillinv",
        "(Pseudonym) History Usecase.xlsx": "df_hist",
        "(Pseudonym) Evaluation Scores.xlsx": "df_eval",
    }
    for f in json_data:
        name, data = f.get("FileName"), f.get("Data")
        key = filename_mapping.get(name)
        if key and data:
            loaded[key] = pd.DataFrame(data)
    return loaded


def load_datasets_from_json_newfile(json_data):
    """Convert uploaded datasets JSON → dictionary of pandas DataFrames."""
    loaded = {}
    filename_mapping = {
        "Usecase Requirement.xlsx": "df_ureq",
        "(Pseudonym) Talent Data.xlsx": "df_talent",
        "(Pseudonym) Self-Assessment Score.xlsx": "df_selfassessment",
        "(Pseudonym) History Usecase.xlsx": "df_hist",
        "(Pseudonym) Capability Scores.xlsx": "df_eval",
        "(Pseudonym) Assignment Data.xlsx": "df_assign"
    }
    for f in json_data:
        name, data = f.get("FileName"), f.get("Data")
        key = filename_mapping.get(name)
        if key and data:
            loaded[key] = pd.DataFrame(data)
    return loaded

def load_final_df_from_blob(url):
    """Fetch df_final (preprocessed) JSON from Azure Blob Storage."""
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.route(route="blob_custom_talent_search_unified", methods=["POST"])
def blob_custom_talent_search_unified(req: func.HttpRequest) -> func.HttpResponse:
    """
    Unified Talent Search:
    - Supports BOTH single and multiple user_input formats.
      user_input = { ... }     → single mode
      user_input = [ {...} ]   → multi mode
    - Runs the same logic for each.
    """

    try:
        logging.info("Starting unified custom talent search...")

        # -------------------------------------------------------
        # 1️⃣ Parse Request
        # -------------------------------------------------------
        body = req.get_json()

        raw_user_input = body.get("user_input")
        datasets_json = body.get("datasets", [])
        blob_url = body.get("blob_url")

        if not blob_url:
            blob_url = "https://azurecleanstorage.blob.core.windows.net/blobcleancontainer/latest.json"

        # Parameters
        threshold = float(body.get("threshold", 0.3))
        top_n = int(body.get("top_n", 10))
        coefficients = body.get("coefficients", {})
        a = float(coefficients.get("a", 0.42))
        b = float(coefficients.get("b", 0.48))
        c = float(coefficients.get("c", 0.1))
        r = float(coefficients.get("r", 1.2))
        bobot_cap_score = float(coefficients.get("bobot_cap_score", 0.8))
        bobot_durasi = float(coefficients.get("bobot_durasi", 0.2))

        # -------------------------------------------------------
        # 2️⃣ Normalize user_input → always becomes a list
        # -------------------------------------------------------
        if raw_user_input is None:
            return func.HttpResponse(
                json.dumps({"error": "'user_input' field is missing"}),
                status_code=400
            )

        if isinstance(raw_user_input, dict):
            user_inputs = [raw_user_input]
            single_mode = True
        elif isinstance(raw_user_input, list):
            if not all(isinstance(x, dict) for x in raw_user_input):
                return func.HttpResponse(
                    json.dumps({"error": "Every element inside user_input must be an object/dict"}),
                    status_code=400
                )
            user_inputs = raw_user_input
            single_mode = False
        else:
            return func.HttpResponse(
                json.dumps({"error": "user_input must be an object or a list of objects"}),
                status_code=400
            )

        # -------------------------------------------------------
        # 3️⃣ Load all datasets
        # -------------------------------------------------------
        datasets = load_datasets_from_json_newfile(datasets_json)

        df_selfassessment = datasets.get("df_selfassessment")
        df_talent = datasets.get("df_talent")
        df_eval = datasets.get("df_eval")
        df_hist = datasets.get("df_hist")
        df_final = load_final_df_from_blob(blob_url)

        if df_selfassessment is None or df_talent is None or df_eval is None:
            return func.HttpResponse(
                json.dumps({"error": "Missing required datasets (selfassessment, talent, eval)."}),
                status_code=400
            )

        model = get_model()

        # -------------------------------------------------------
        # 4️⃣ Define reusable processing function for 1 input
        # -------------------------------------------------------

        def process_one_input(user_input):
            # Extract fields
            responsibility = user_input.get("responsibility", "")
            skill1 = user_input.get("skill1", "")
            skill2 = user_input.get("skill2", "")
            role = user_input.get("role", "")
            job_level = user_input.get("job_level", 0)

            query_text = f"{responsibility} {skill1} {skill2} {role}"

            # --- Compute similarity
            skillsets = [c for c in df_selfassessment.columns if c not in ["UNIQUE ID"]]
            corpus = [query_text] + skillsets
            embeddings = model.encode(corpus)
            cosine_sim = cosine_similarity([embeddings[0]], embeddings[1:])[0]

            df_similarity = pd.DataFrame({
                "Skillset": skillsets,
                "Similarity": cosine_sim
            })
            df_similarity = df_similarity[df_similarity["Similarity"] >= threshold]

            if df_similarity.empty:
                return {
                    "metadata": {
                        "query": user_input,
                        "message": "No matching skills found above threshold."
                    },
                    "results": []
                }

            # Cross join IDs
            unique_ids = df_selfassessment[["UNIQUE ID"]].drop_duplicates()
            merged = df_similarity.merge(unique_ids, how="cross")

            def get_skill_score(row):
                try:
                    val = df_selfassessment.loc[
                        df_selfassessment["UNIQUE ID"] == row["UNIQUE ID"], row["Skillset"]
                    ]
                    return float(val.values[0]) if not val.empty else np.nan
                except:
                    return np.nan

            merged["Skill Score"] = merged.apply(get_skill_score, axis=1)

            # Aggregate
            df_search = merged.groupby("UNIQUE ID", as_index=False).agg(
                Avg_SkillScore=("Skill Score", "mean")
            )

            # Merge additional datasets
            df_merged = (
                df_search
                .merge(df_talent, on="UNIQUE ID", how="left")
                .merge(df_eval, on="UNIQUE ID", how="left")
                .merge(df_final, on="UNIQUE ID", how="left", suffixes=("", "_final"))
            )

            for col in ["Role", "Responsibilities"]:
                if f"{col}_final" in df_merged.columns:
                    df_merged[col] = df_merged[f"{col}_final"]

            df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith("_final")]
            
            if "Job_Level" in df_merged.columns:
                df_merged["Job_Level"] = pd.to_numeric(
                    df_merged["Job_Level"], errors="coerce"
                ).fillna(0)
            else:
                df_merged["Job_Level"] = 0

            # Get requested job_level from user input
            requested_job_level = int(job_level) if job_level is not None else 0

            # Filter talents: only allow equal or higher level
            df_merged = df_merged[
                df_merged["Job_Level"] >= requested_job_level
            ]

            # job_count from history
            if df_hist is not None and "UNIQUE ID" in df_hist.columns:
                job_counts = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique()
                df_merged["job_count"] = df_merged["UNIQUE ID"].map(job_counts).fillna(0)
            else:
                df_merged["job_count"] = 0

            # Weighted scoring
            df_merged["Avg_SkillScore"] = df_merged["Avg_SkillScore"].fillna(0)
            df_merged["Capability Score"] = df_merged["Capability Score"].fillna(0)

            df_merged["d"] = (
                df_merged["Avg_SkillScore"] * a +
                df_merged["Capability Score"] * b +
                df_merged["job_count"] * c
            )

            def apply_role_bonus(row):
                role_final = str(row.get("Role", "")).strip().lower()
                return row["d"] * r if role_final == role.lower().strip() else row["d"]

            df_merged["finalscore"] = df_merged.apply(apply_role_bonus, axis=1)
            df_merged["finalscore_scaled"] = minmax_scaling(df_merged["finalscore"])

            df_ranked = (
                df_merged.sort_values("finalscore_scaled", ascending=False)
                .drop_duplicates(subset=["UNIQUE ID"], keep="first")
            )

            results = json.loads(df_ranked.to_json(orient="records"))
            results = results[:top_n]

            return {
                "metadata": {
                    "query": user_input,
                    "total_candidates": len(df_ranked),
                    "returned_candidates": len(results)
                },
                "results": results
            }

        # -------------------------------------------------------
        # 5️⃣ Process each input
        # -------------------------------------------------------
        output_list = [process_one_input(u) for u in user_inputs]

        # Return single or list depending on input type
        final_output = output_list[0] if single_mode else output_list

        return func.HttpResponse(
            json.dumps(final_output, indent=2),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error in unified talent search: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}), status_code=500
        )
