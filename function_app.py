import azure.functions as func
import logging
import base64
from io import BytesIO
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
from sentence_transformers import SentenceTransformer

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
    
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_datasets_from_local(base_path="dataset"):
    """Loads all required Excel datasets into a global dictionary."""
    global DATASETS
    if DATASETS:
        return DATASETS
    
    dataset_files = {
        "df_ureq": "Usecase Requirement.xlsx",
        "df_talent": "(Pseudonym) Talent Data.xlsx",
        "df_skillinv": "(Pseudonym) Skill Inventory.xlsx",
        "df_hist": "(Pseudonym) History Usecase.xlsx",
        "df_eval": "(Pseudonym) Evaluation Scores.xlsx",
        "df_assign": "(Pseudonym) Assignment Data.xlsx"
    }
    
    loaded_datasets = {}
    try:
        for key, filename in dataset_files.items():
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                loaded_datasets[key] = pd.read_excel(file_path)
                logging.info(f"Successfully loaded {filename}")
            else:
                raise FileNotFoundError(f"Dataset file not found at {file_path}")
        DATASETS = loaded_datasets
        return DATASETS
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return None

def load_datasets_from_json(json_data):
    """Loads datasets from JSON data received in the request body."""
    global DATASETS
    
    DATASETS = {}

    try:
        loaded_datasets = {}
        
        # Map of expected filenames to dataset keys
        filename_mapping = {
            "Usecase Requirement.xlsx": "df_ureq",
            "(Pseudonym) Talent Data.xlsx": "df_talent",
            "(Pseudonym) Skill Inventory.xlsx": "df_skillinv",
            "(Pseudonym) History Usecase.xlsx": "df_hist",
            "(Pseudonym) Evaluation Scores.xlsx": "df_eval",
            "(Pseudonym) Assignment Data.xlsx": "df_assign"
        }
        
        # Process each file in the JSON array
        for file_obj in json_data:
            filename = file_obj.get("FileName")
            data = file_obj.get("Data")
            
            if not filename or data is None:
                logging.warning(f"Skipping invalid entry: {file_obj}")
                continue
            
            # Find the corresponding dataset key
            dataset_key = filename_mapping.get(filename)
            
            if dataset_key:
                # Convert the data array to a pandas DataFrame
                df = pd.DataFrame(data)
                loaded_datasets[dataset_key] = df
                logging.info(f"Successfully loaded {filename} with {len(df)} rows")
            else:
                logging.warning(f"Unknown filename: {filename}")
        
        # Check if all required datasets were loaded
        required_keys = set(filename_mapping.values())
        loaded_keys = set(loaded_datasets.keys())
        missing_keys = required_keys - loaded_keys
        
        if missing_keys:
            logging.warning(f"Missing datasets: {missing_keys}")
        
        DATASETS = loaded_datasets
        return DATASETS
        
    except Exception as e:
        logging.error(f"Error loading datasets from JSON: {e}")
        return None

# --- HELPER & PREPROCESSING FUNCTIONS ---
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

def minmax_scaling(series):
    if series.max() == series.min(): return 0.5
    return (series - series.min()) / (series.max() - series.min())

@app.route(route="custom_talent_search")
def custom_talent_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    Custom Talent Search - User defines their own job requirements
    
    Endpoint: /api/custom_talent_search
    Method: POST
    
    Request Body:
    {
        "datasets": [...],  // Optional for Power Automate, omit for local testing
        "user_input": {
            "responsibility": "Job description text",
            "skill1": "Primary skill",
            "skill2": "Secondary skill",
            "role": "Job role",
            "job_level": 3  // 1-5, optional
        },
        "threshold": 0.3,  // Optional, default 0.3
        "coefficients": {  // Optional, defaults provided
            "a": 0.42,
            "b": 0.48,
            "c": 0.1,
            "r": 1.2,
            "bobot_cap_score": 0.8,
            "bobot_durasi": 0.2
        },
        "top_n": 10  // Optional, number of results to return
    }
    """
    logging.info('Custom Talent Search function processing a request.')
    
    try:
        # Parse request body
        try:
            req_body = req.get_json() if req.get_body() else {}
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400
            )
        
        # Load datasets (from JSON or local files)
        datasets_json = req_body.get('datasets')
        if datasets_json:
            datasets = load_datasets_from_json(datasets_json)
        else:
            datasets = load_datasets_from_local()
        
        if not datasets:
            return func.HttpResponse(
                json.dumps({"error": "Could not load datasets. Check logs."}),
                status_code=500
            )
        
        # Extract user input parameters
        user_input = req_body.get('user_input', {})
        user_responsibility = user_input.get('responsibility')
        user_skill1 = user_input.get('skill1', '')
        user_skill2 = user_input.get('skill2', '')
        user_role = user_input.get('role')
        user_job_level = int(user_input.get('job_level', 0))
        
        # Extract configuration
        threshold = float(req_body.get('threshold', 0.3))
        top_n = int(req_body.get('top_n', 10))
        
        coefficients = req_body.get('coefficients', {})
        a = float(coefficients.get('a', 0.42))
        b = float(coefficients.get('b', 0.48))
        c = float(coefficients.get('c', 0.1))
        r = float(coefficients.get('r', 1.2))
        bobot_cap_score = float(coefficients.get('bobot_cap_score', 0.8))
        bobot_durasi = float(coefficients.get('bobot_durasi', 0.2))
        
        # Validate required inputs
        if not user_responsibility or not user_role:
            return func.HttpResponse(
                json.dumps({
                    "error": "Missing required fields in 'user_input': 'responsibility' and 'role' are required."
                }),
                status_code=400
            )
        
        logging.info(f"Searching for talent with responsibility: {user_responsibility}, role: {user_role}")
        
        # STEP 1: Create user requirement DataFrame
        df_user_ureq = pd.DataFrame({
            'Responsibilities': [user_responsibility],
            'Skill 1': [user_skill1],
            'Skill 2': [user_skill2],
            'Role': [user_role]
        })
        df_user_ureq['agg_sentences'] = (
            df_user_ureq['Responsibilities'] + " " + 
            df_user_ureq['Skill 1'].fillna('') + " " + 
            df_user_ureq['Skill 2'].fillna('')
        )
        
        # STEP 2: Calculate similarity between user input and available skillsets
        sentence_model = get_model()
        df_skillinv = datasets['df_skillinv'].copy()
        skillsets = df_skillinv.columns[2:-1].tolist()
        
        user_agg_sentence = df_user_ureq['agg_sentences'].iloc[0]
        corpus_user = [user_agg_sentence] + skillsets
        sentence_embeddings_user = sentence_model.encode(corpus_user)
        cosine_sim_user = cosine_similarity([sentence_embeddings_user[0]], sentence_embeddings_user[1:])
        
        # Build similarity results
        results_user = []
        for i in range(len(skillsets)):
            results_user.append([
                user_responsibility, user_skill1, user_skill2,
                user_role, skillsets[i], cosine_sim_user[0][i]
            ])
        
        df_results_user = pd.DataFrame(results_user, columns=[
            'Responsibilities', 'Skill 1', 'Skill 2', 'Role', 'Skillset', 'Similarity score'
        ])
        
        # Filter by threshold
        df_results_user_filtered = df_results_user[df_results_user['Similarity score'] >= threshold]
        
        if df_results_user_filtered.empty:
            return func.HttpResponse(
                json.dumps({
                    "message": f"No matching skillsets found with threshold {threshold}. Try lowering the threshold.",
                    "suggested_threshold": 0.2,
                    "results": []
                }),
                status_code=200,
                mimetype="application/json"
            )
        
        logging.info(f"Found {len(df_results_user_filtered)} matching skillsets above threshold {threshold}")
        
        # STEP 3: Cross-join with talent pool
        unique_ids_roles = df_skillinv[['UNIQUE ID', 'Role']].drop_duplicates()
        unique_ids_roles = unique_ids_roles.rename(columns={'Role': 'Role Person'})
        merged_df_user = df_results_user_filtered.merge(unique_ids_roles, how='cross')
        
        # Get skill scores for each talent
        def get_skill_score_user(row):
            try:
                score = df_skillinv.loc[
                    df_skillinv['UNIQUE ID'] == row['UNIQUE ID'], 
                    row['Skillset']
                ]
                return float(score.values[0]) if not score.empty else np.nan
            except Exception:
                return np.nan
        
        merged_df_user['Skill Score'] = merged_df_user.apply(get_skill_score_user, axis=1)
        
        # Aggregate by talent
        df_search_user = merged_df_user.groupby(
            ['Responsibilities', 'Role', 'UNIQUE ID', 'Role Person'],
            as_index=False
        ).agg(Avg_SkillScore=('Skill Score', 'mean'))
        
        # STEP 4: Add clustering information
        df_numerical = df_skillinv.select_dtypes(include=[np.number]).dropna(axis=1)
        if len(df_numerical) >= 4 and len(df_numerical.columns) >= 2:
            pca = PCA(n_components=2)
            df_pca = pca.fit_transform(df_numerical)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
            df_skillinv['Cluster'] = kmeans.fit_predict(df_pca)
            df_search_user = df_search_user.merge(
                df_skillinv[['UNIQUE ID', 'Cluster']], 
                on='UNIQUE ID', 
                how='left'
            )
        else:
            df_search_user['Cluster'] = -1
        
        # STEP 5: Merge with talent data and evaluation scores
        df_talent = datasets['df_talent'].copy()
        df_eval = datasets['df_eval'].copy()
        
        # Convert work duration to months
        if 'LAMA KERJA BERJALAN' in df_talent.columns:
            df_talent['Durasi Bulan'] = df_talent['LAMA KERJA BERJALAN'].apply(convert_to_months)
        else:
            df_talent['Durasi Bulan'] = 0
        
        df_agg_talent = pd.merge(df_talent, df_eval, on='UNIQUE ID', how='inner')
        
        # Calculate evaluation score
        if 'Durasi Bulan' in df_agg_talent.columns and 'Capability Score' in df_agg_talent.columns:
            df_agg_talent['scoring_eval'] = (
                df_agg_talent['Durasi Bulan'] * bobot_durasi +
                df_agg_talent['Capability Score'] * bobot_cap_score
            )
        else:
            df_agg_talent['scoring_eval'] = 0
        
        df_merged = pd.merge(df_search_user, df_agg_talent, on='UNIQUE ID', how='inner')
        
        # STEP 6: Add job count from history
        df_hist = datasets['df_hist'].copy()
        df_hist_count = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique().reset_index(name="job_count")
        df_final = pd.merge(df_merged, df_hist_count, on='UNIQUE ID', how='left').fillna({'job_count': 0})
        
        # STEP 7: Calculate final scores
        df_final['d'] = (
            df_final['Avg_SkillScore'] * a +
            df_final['scoring_eval'] * b +
            df_final['job_count'] * c
        )
        
        # Apply role matching bonus
        df_final['finalscore'] = df_final.apply(
            lambda row: row['d'] * r if row['Role Person'] == row['Role'] else row['d'], 
            axis=1
        )
        
        # Normalize scores
        df_final['finalscore_scaled'] = df_final.groupby('Responsibilities')['finalscore'].transform(minmax_scaling)
        
        # Sort by final score
        df_final = df_final.sort_values(by='finalscore_scaled', ascending=False)
        
        # Limit to top N results
        df_final_top = df_final.head(top_n)
        
        # STEP 8: Format response
        response_cols = [
            'UNIQUE ID', 'Role Person', 'Responsibilities', 
            'finalscore_scaled', 'finalscore', 'Avg_SkillScore', 
            'scoring_eval', 'job_count', 'Cluster'
        ]
        response_cols = [col for col in response_cols if col in df_final_top.columns]
        
        # Prepare metadata
        metadata = {
            "query": {
                "responsibility": user_responsibility,
                "skill1": user_skill1,
                "skill2": user_skill2,
                "role": user_role,
                "job_level": user_job_level
            },
            "parameters": {
                "threshold": threshold,
                "top_n": top_n,
                "coefficients": {
                    "a": a, "b": b, "c": c, "r": r,
                    "bobot_cap_score": bobot_cap_score,
                    "bobot_durasi": bobot_durasi
                }
            },
            "total_candidates": len(df_final),
            "returned_candidates": len(df_final_top)
        }
        
        # Build final response
        response = {
            "metadata": metadata,
            "results": json.loads(df_final_top[response_cols].to_json(orient="records"))
        }
        
        return func.HttpResponse(
            json.dumps(response, indent=2),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"An error occurred in custom_talent_search: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500
        )