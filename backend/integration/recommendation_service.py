
import os, requests, joblib, json, datetime
from urllib.parse import urljoin
import numpy as np
import pandas as pd

# ------------------- CONFIGURATION -------------------
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json'
}

MODEL_PATH = 'Python_Model/model.pkl'
LOCAL_SCHOLARSHIPS_CSV = 'Python_Model/scholarships.csv'

# ------------------- PROFIL UTILISATEUR -------------------
def fetch_profile(user_id):
    """Récupère le profil complet d'un utilisateur depuis Supabase"""
    if not SUPABASE_URL:
        raise EnvironmentError('SUPABASE_URL not set')

    url = urljoin(SUPABASE_URL, f"/rest/v1/profiles?user_id=eq.{user_id}&select=*")
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else {}

# ------------------- BOURSES -------------------
def fetch_scholarships():
    """Récupère toutes les bourses depuis Supabase ou fallback CSV"""
    if SUPABASE_URL:
        try:
            url = urljoin(SUPABASE_URL, '/rest/v1/scholarships?select=*')
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass

    if os.path.exists(LOCAL_SCHOLARSHIPS_CSV):
        df = pd.read_csv(LOCAL_SCHOLARSHIPS_CSV)
        return df.fillna('').to_dict(orient='records')

    return []

# ------------------- ML PREDICTIONS -------------------
def predict_with_model(profile, scholarships):
    """Score ML basé sur model.pkl; fallback heuristique complète"""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            rows = []

            for s in scholarships:
                amt = float(s.get('montant') or 0)

                country_match = 1.0 if profile.get('pays_origine') and profile['pays_origine'].lower() in (s.get('pays') or '').lower() else 0.0
                target_country_match = 1.0 if profile.get('pays_cible') and profile['pays_cible'].lower() in (s.get('pays') or '').lower() else 0.0
                domain_match = 1.0 if profile.get('domaine_etude') and profile['domaine_etude'].lower() in (s.get('domaine') or '').lower() else 0.0
                level_match = 1.0 if profile.get('niveau_etude') and profile['niveau_etude'].lower() in (s.get('niveau_etudes') or '').lower() else 0.0
                mention_match = 1.0 if profile.get('mention_scolaire') and str(profile['mention_scolaire']).lower() in str(s.get('mentions_min', '')).lower() else 0.0
                type_match = 1.0 if profile.get('type_bourse') and profile['type_bourse'].lower() in (s.get('type') or '').lower() else 0.0
                financement_match = 1.0 if profile.get('type_financement') and profile['type_financement'].lower() in (s.get('type_financement') or '').lower() else 0.0

                # Âge
                age_match = 1.0
                try:
                    age = int(profile.get('âge') or profile.get('age') or 0)
                    age_min = int(s.get('age_min') or 0)
                    age_max = int(s.get('age_max') or 100)
                    if age < age_min or age > age_max:
                        age_match = 0.0
                except:
                    pass

                rows.append([
                    amt, country_match, target_country_match, domain_match,
                    level_match, mention_match, type_match, financement_match, age_match
                ])

            X = np.array(rows)
            preds = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else model.predict(X)

            # Normalisation 0-1
            minv, maxv = preds.min(), preds.max()
            if maxv - minv > 0:
                preds = (preds - minv) / (maxv - minv)
            return preds.tolist()

        except Exception as e:
            print("Model load/predict failed:", e)

    # ---------- Fallback heuristique ----------
    scores = []
    for s in scholarships:
        score = 0.0

        # 1️⃣ Pays d’origine
        if profile.get('pays_origine') and profile['pays_origine'].lower() in (s.get('pays') or '').lower():
            score += 0.1

        # 2️⃣ Pays cible
        if profile.get('pays_cible') and profile['pays_cible'].lower() in (s.get('pays') or '').lower():
            score += 0.1

        # 3️⃣ Domaine d’étude
        if profile.get('domaine_etude') and profile['domaine_etude'].lower() in (s.get('domaine') or '').lower():
            score += 0.2

        # 4️⃣ Niveau d’étude
        if profile.get('niveau_etude') and profile['niveau_etude'].lower() in (s.get('niveau_etudes') or '').lower():
            score += 0.15

        # 5️⃣ Mention scolaire
        try:
            mention = float(profile.get('mention_scolaire') or 0)
            min_mentions = float(s.get('mentions_min') or 0)
            if mention >= min_mentions:
                score += 0.1
            else:
                score -= 0.05
        except:
            pass

        # 6️⃣ Type de bourse
        if profile.get('type_bourse') and profile['type_bourse'].lower() in (s.get('type') or '').lower():
            score += 0.1

        # 7️⃣ Type de financement
        if profile.get('type_financement') and profile['type_financement'].lower() in (s.get('type_financement') or '').lower():
            score += 0.1

        # 8️⃣ Montant
        try:
            amt = float(s.get('montant') or 0)
            score += min(amt / 10000.0, 0.05)
        except:
            pass

        # 9️⃣ Âge
        try:
            age = int(profile.get('âge') or profile.get('age') or 0)
            age_min = int(s.get('age_min') or 0)
            age_max = int(s.get('age_max') or 100)
            if age < age_min or age > age_max:
                score -= 0.1
        except:
            pass

        scores.append(score)

    arr = np.array(scores)
    if arr.max() - arr.min() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr.tolist()

# ------------------- RÈGLES MÉTIER -------------------
def apply_rule_based_filters(profile, scholarships):
    """Boosts/penalités selon les préférences utilisateur"""
    results = []

    for s in scholarships:
        score = 0.0

        # 🌍 Préférence de pays cible
        if profile.get('pays_cible') and profile['pays_cible'].lower() in (s.get('pays') or '').lower():
            score += 0.1

        # 🎓 Domaine cohérent
        if profile.get('domaine_etude') and profile['domaine_etude'].lower() in (s.get('domaine') or '').lower():
            score += 0.1

        # 💰 Type de financement
        if profile.get('type_financement') and profile['type_financement'].lower() in (s.get('type_financement') or '').lower():
            score += 0.1

        # 🧾 Mention scolaire minimale
        try:
            mention = float(profile.get('mention_scolaire') or 0)
            min_mentions = float(s.get('mentions_min') or 0)
            if mention < min_mentions:
                score -= 0.1
        except:
            pass

        # 🧑‍🎓 Âge admissible
        try:
            age = int(profile.get('âge') or profile.get('age') or 0)
            age_min = int(s.get('age_min') or 0)
            age_max = int(s.get('age_max') or 100)
            if age < age_min or age > age_max:
                score -= 0.1
        except:
            pass

        results.append(score)

    return results

# ------------------- ÉCRITURE RECOMMANDATIONS -------------------
def write_recommendations(user_id, recs):
    """Écrit les recommandations dans Supabase"""
    if not SUPABASE_URL:
        raise EnvironmentError('SUPABASE_URL not set')

    url = urljoin(SUPABASE_URL, '/rest/v1/recommendations')
    payload = []

    for s_id, score in recs:
        payload.append({
            'user_id': user_id,
            'scholarship_id': s_id,
            'score': score,
            'created_at': datetime.datetime.utcnow().isoformat()
        })

    r = requests.post(url, headers=HEADERS, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()
