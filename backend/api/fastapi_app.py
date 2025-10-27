from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import recommendation_service as rs

app = FastAPI(title="EduGrant Finder API", version="2.0")

# ------------------- 🧾 Modèle de profil utilisateur -------------------
class UserProfile(BaseModel):
    nom_complet: Optional[str] = Field(None, description="Nom complet de l'utilisateur")
    age: Optional[int] = Field(None, description="Âge de l'utilisateur")
    pays_origine: Optional[str] = Field(None, description="Pays d'origine")
    pays_cible: Optional[str] = Field(None, description="Pays où il souhaite étudier")
    domaine_etude: Optional[str] = Field(None, description="Domaine d'étude")
    niveau_etude: Optional[str] = Field(None, description="Niveau d'étude actuel ou visé")
    mention_scolaire: Optional[float] = Field(None, description="Note ou mention scolaire")
    type_bourse: Optional[str] = Field(None, description="Type de bourse souhaité (ex: académique, sportive, humanitaire...)")
    type_financement: Optional[str] = Field(None, description="Type de financement souhaité (ex: complète, partielle, logement...)")

# ------------------- 📩 Modèle de requête -------------------
class PredictRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Identifiant utilisateur si enregistré en base")
    profile: Optional[UserProfile] = Field(None, description="Profil complet si non connecté")
    limit: int = Field(10, description="Nombre maximum de bourses recommandées à retourner")

# ------------------- 🎯 Endpoint principal -------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # 1️⃣ Récupération de la liste des bourses
        scholarships = rs.fetch_scholarships()
        if not scholarships:
            raise HTTPException(status_code=404, detail="Aucune bourse disponible")

        # 2️⃣ Récupération du profil utilisateur
        if req.user_id:
            profile = rs.fetch_profile(req.user_id)
        else:
            profile = req.profile.dict() if req.profile else {}

        if not profile:
            raise HTTPException(status_code=400, detail="Profil utilisateur manquant")

        # 3️⃣ Prédictions via modèle et règles
        ml_scores = rs.predict_with_model(profile, scholarships)
        rule_scores = rs.apply_rule_based_filters(profile, scholarships)

        # 4️⃣ Fusion pondérée (70% ML / 30% règles)
        combined = []
        for i, s in enumerate(scholarships):
            score = (0.7 * ml_scores[i]) + (0.3 * rule_scores[i])
            combined.append((s.get('id'), float(score), s))

        # 5️⃣ Tri + limitation
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:req.limit]

        # 6️⃣ Écriture dans Supabase si user_id fourni
        if req.user_id:
            rs.write_recommendations(req.user_id, [(item[0], item[1]) for item in combined_sorted])

        # 7️⃣ Retour JSON structuré
        return {
            "user_id": req.user_id,
            "count": len(combined_sorted),
            "recommendations": [
                {
                    "scholarship_id": item[0],
                    "score": round(item[1], 4),
                    "scholarship": item[2]
                } for item in combined_sorted
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# ------------------- 🌐 Endpoint de test rapide -------------------
@app.get("/")
def root():
    return {"message": "✅ EduGrant Finder API opérationnelle", "version": "2.0"}
