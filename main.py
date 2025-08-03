# main.py
import os
from google.cloud import bigquery

# ✅ Set the environment variable explicitly (REQUIRED for Render)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/tourismrecommender-93df7478854c.json"

# ✅ Now GCP client will find the credentials
client = bigquery.Client()

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Optional

app = FastAPI()

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ──────────────────────────────────────────────────────────────────────────────

# 1) LOAD & PREP DATA ─────────────────────────────────────────────────────────
query = """
SELECT id, name, address, city, state, zipcode, rating,
       reviews, categories, broader_category, Weighted_Score, lat, lon
FROM `tourismrecommender.tourism_data.tourist_places`
WHERE lat IS NOT NULL AND lon IS NOT NULL
"""
df = client.query(query).to_dataframe()

# Build richer text for embeddings
df['embed_text'] = (
    df[['name', 'categories', 'city', 'state', 'address']]
    .astype(str)
    .agg(' '.join, axis=1)
)

# Pre-extract numpy arrays for performance
LAT_ARR = df['lat'].values
LON_ARR = df['lon'].values
RATING_ARR = df['rating'].values
CATS = df['broader_category'].fillna('Other').values

# 2) PRECOMPUTE EMBEDDINGS ────────────────────────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')
PLACE_EMBS = model.encode(
    df['embed_text'].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

# 3) VECTORIZED HAVERSINE ─────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# 4) RECOMMEND ENDPOINT ───────────────────────────────────────────────────────
@app.get("/recommend")
def recommend(
    lat: float             = Query(..., ge=-90,  le=90),
    lon: float             = Query(..., ge=-180, le=180),
    top_k: int             = Query(5,   ge=1),
    radius_km: float       = Query(500, ge=0.0),
    interest: Optional[str] = Query(None, description="Optional text bias")
):
    """
    Recommend up to `top_k` places within `radius_km` of (lat, lon).
    Bias by `interest` if provided, blending semantic similarity,
    distance decay, rating boost, and category diversity.
    """
    try:
        # distances
        dists = haversine(lat, lon, LAT_ARR, LON_ARR)

        # filter by radius
        mask = dists <= radius_km
        if not mask.any():
            return {"places": []}

        cand_idxs   = np.where(mask)[0]
        cand_dists  = dists[mask]
        cand_ratings= RATING_ARR[mask]
        cand_cats   = CATS[mask]

        # semantic similarity
        if interest:
            user_emb = model.encode(interest, convert_to_tensor=True)
            sims = util.cos_sim(user_emb, PLACE_EMBS[cand_idxs])[0].cpu().numpy()
        else:
            sims = np.ones_like(cand_dists)

        # dynamic decay
        sigma = radius_km / 2.0 if radius_km < 100 else radius_km / 3.0
        decay = np.exp(-cand_dists / sigma)

        # rating boost
        alpha = 0.2
        rating_boost = np.power(cand_ratings / 5.0, alpha)

        # final score
        scores = sims * decay * rating_boost

        # build candidate DataFrame
        cand_df = pd.DataFrame({
            'idx':         cand_idxs,
            'distance_km': cand_dists,
            'rating':      cand_ratings,
            'score':       scores,
            'category':    cand_cats
        })

        # diversify: pick top_k*3 then round-robin by category
        top_scored = cand_df.sort_values(by='rating', ascending=False).head(top_k * 3)
        buckets = {
            cat: grp.sort_values('score', ascending=False).to_dict('records')
            for cat, grp in top_scored.groupby('category')
        }
        final = []
        cats_cycle = list(buckets.keys())
        while len(final) < top_k and cats_cycle:
            for cat in list(cats_cycle):
                if buckets[cat]:
                    final.append(buckets[cat].pop(0))
                    if len(final) == top_k:
                        break
                else:
                    cats_cycle.remove(cat)

        # sort nearest-first within final
        final_sorted = sorted(final, key=lambda x: x['distance_km'])

        # map back and include lat/lon for directions
        records = []
        for row in final_sorted:
            src = df.iloc[int(row['idx'])]
            records.append({
                'name':        src['name'],
                'address':     src['address'],
                'city':        src['city'],
                'state':       src['state'],
                'zipcode':     src['zipcode'],
                'rating':      src['rating'],
                'distance_km': row['distance_km'],
                'lat':         src['lat'],
                'lon':         src['lon'],
            })

        return {"places": records}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# 5) SEARCH ENDPOINT ─────────────────────────────────────────────────────────
@app.get("/search")
def search_places(query: str, top_n: int = Query(5, ge=1)):
    from fuzzywuzzy import process
    names = df['name'].fillna('').tolist()
    results = process.extract(query, names, limit=top_n)
    matches = [n for (n, score) in results if score > 70]
    return {"matches": matches}


# 6) CATEGORIES ENDPOINT ──────────────────────────────────────────────────────
@app.get("/categories")
def list_categories():
    """Return all distinct broader_category values."""
    cats = df['broader_category'].dropna().unique().tolist()
    return {"categories": sorted(cats)}
