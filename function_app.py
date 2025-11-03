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

def load_final_df_from_blob(url):
    """Fetch df_final (preprocessed) JSON from Azure Blob Storage."""
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.route(route="blob_custom_talent_search", methods=["POST"])
def blob_custom_talent_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    Hybrid Custom Talent Search:
    - Loads df_final from Azure Blob
    - Loads other datasets (skillinv, eval, etc.) from JSON input
    - Computes similarity and scores dynamically
    """

    try:
        logging.info("Starting hybrid_custom_talent_search...")

        # 1️⃣ Parse request
        body = req.get_json()
        user_input = body.get("user_input", {})
        datasets_json = body.get("datasets", [])
        blob_url = body.get("blob_url", "https://azurecleanstorage.blob.core.windows.net/blobcleancontainer/latest.json")

        threshold = float(body.get("threshold", 0.3))
        top_n = int(body.get("top_n", 10))
        coefficients = body.get("coefficients", {})
        a = float(coefficients.get("a", 0.42))
        b = float(coefficients.get("b", 0.48))
        c = float(coefficients.get("c", 0.1))
        r = float(coefficients.get("r", 1.2))
        bobot_cap_score = float(coefficients.get("bobot_cap_score", 0.8))
        bobot_durasi = float(coefficients.get("bobot_durasi", 0.2))

        # 2️⃣ Load datasets
        datasets = load_datasets_from_json(datasets_json)
        df_skillinv = datasets.get("df_skillinv")
        df_talent = datasets.get("df_talent")
        df_eval = datasets.get("df_eval")
        df_hist = datasets.get("df_hist")
        df_final = load_final_df_from_blob(blob_url)

        if df_skillinv is None or df_talent is None or df_eval is None:
            return func.HttpResponse(
                json.dumps({"error": "Missing required datasets (skillinv, talent, eval)."}), status_code=400
            )

        # 3️⃣ Build query sentence
        responsibility = user_input.get("responsibility", "")
        skill1 = user_input.get("skill1", "")
        skill2 = user_input.get("skill2", "")
        role = user_input.get("role", "")
        job_level = user_input.get("job_level", 0)
        query_text = f"{responsibility} {skill1} {skill2} {role}"

        model = get_model()

        # 4️⃣ Compute similarity with skill inventory
        skillsets = df_skillinv.columns[2:-1].tolist()
        corpus = [query_text] + skillsets
        embeddings = model.encode(corpus)
        cosine_sim = cosine_similarity([embeddings[0]], embeddings[1:])[0]

        df_similarity = pd.DataFrame({
            "Skillset": skillsets,
            "Similarity": cosine_sim
        })
        df_similarity = df_similarity[df_similarity["Similarity"] >= threshold]

        if df_similarity.empty:
            return func.HttpResponse(
                json.dumps({"message": "No matching skills found above threshold."}),
                status_code=200
            )

        # 5️⃣ Cross join with UNIQUE IDs and calculate skill scores
        unique_roles = df_skillinv[["UNIQUE ID", "Role"]].rename(columns={"Role": "Role Person"}).drop_duplicates()
        merged = df_similarity.merge(unique_roles, how="cross")

        def get_skill_score(row):
            try:
                val = df_skillinv.loc[df_skillinv["UNIQUE ID"] == row["UNIQUE ID"], row["Skillset"]]
                return float(val.values[0]) if not val.empty else np.nan
            except Exception:
                return np.nan

        merged["Skill Score"] = merged.apply(get_skill_score, axis=1)
        # ✅ Normalize roles to ensure match with query
        role_normalization = {
            "Artificial Intelligence Engineer": "AI Engineer",
            "Artificial Intelligence Specialist": "AI Engineer",
            "Data Analyst": "Data Analyst",  # example, add if more exist
        }
        df_skillinv["Role"] = df_skillinv["Role"].replace(role_normalization)
        df_final["Role Person"] = df_final["Role Person"].replace(role_normalization)

        # ✅ Aggregate skill similarity per candidate BEFORE merge (original logic)
        df_search = merged.groupby(
            ["UNIQUE ID", "Role Person"], as_index=False
        ).agg(
            Avg_SkillScore=("Skill Score", "mean")
        )

        # ✅ Merge with talent, eval, df_final
        df_merged = (
            df_search
            .merge(df_talent, on="UNIQUE ID", how="left")
            .merge(df_eval, on="UNIQUE ID", how="left")
            .merge(df_final, on="UNIQUE ID", how="left", suffixes=("", "_final"))
        )

        # ✅ Drop duplicate role columns from merge
        for col in ["Role Person", "Role", "Responsibilities"]:
            if f"{col}_final" in df_merged.columns:
                df_merged[col] = df_merged[f"{col}_final"]

        df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith("_final")]

        # ✅ Job Count (if history exists)
        if df_hist is not None and "UNIQUE ID" in df_hist.columns:
            job_counts = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique()
            df_merged["job_count"] = df_merged["UNIQUE ID"].map(job_counts).fillna(0)
        else:
            df_merged["job_count"] = 0

        # ✅ Ensure key fields exist
        df_merged["Avg_SkillScore"] = df_merged["Avg_SkillScore"].fillna(0)
        df_merged["scoring_eval"] = df_merged["scoring_eval"].fillna(0)

        # ✅ Weighted score (d)
        df_merged["d"] = (
            df_merged["Avg_SkillScore"] * a +
            df_merged["scoring_eval"] * b +
            df_merged["job_count"] * c
        )

        # ✅ Apply bonus multiplier for correct role match
        df_merged["finalscore"] = df_merged.apply(
            lambda row: row["d"] * r if str(row["Role Person"]).strip().lower() == str(role).strip().lower() else row["d"],
            axis=1
        )

        # ✅ Scale
        df_merged["finalscore_scaled"] = minmax_scaling(df_merged["finalscore"])

        # ✅ Final unique ranking — just like merged_results
        df_ranked_full = (
            df_merged
            .sort_values("finalscore_scaled", ascending=False)
            .drop_duplicates(subset=["UNIQUE ID"], keep="first")
        )

        # This is your full ranked list
        results = json.loads(df_ranked_full.to_json(orient="records"))

        
        # --- CLEAN COLUMN NAME COLLISIONS ---
        for col in ["Role Person", "Role", "Responsibilities"]:
            if f"{col}_x" in df_merged.columns:
                df_merged[col] = df_merged[f"{col}_x"]
            elif f"{col}_y" in df_merged.columns:
                df_merged[col] = df_merged[f"{col}_y"]

        df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith(("_x", "_y"))]

        # --- DROP DUPLICATE UNIQUE IDs (keep strongest match) ---
        df_merged = df_merged.sort_values("finalscore_scaled", ascending=False)
        df_merged = df_merged.drop_duplicates(subset=["UNIQUE ID"], keep="first")

        # ✅ Final response
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "metadata": {
                    "query": user_input,
                    "source_blob": blob_url,
                    "total_candidates": len(df_ranked_full),
                    "returned_candidates": len(df_ranked_full),
                },
                "results": results  # FULL ranked list
            }, indent=2),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error in hybrid_custom_talent_search: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}), status_code=500
        )
