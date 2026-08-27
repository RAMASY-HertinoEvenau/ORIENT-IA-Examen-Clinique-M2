"""Serveur Web pour ORIENT'IA avec support FastAPI et fallback universel Python natif (http.server)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from orient_ia.agent.orchestrateur import AgentOrientIA

# Instanciation de l'agent unique traçable
agent = AgentOrientIA()


def traiter_analyse_profil(profil_dict: Dict[str, Any]) -> Dict[str, Any]:
    niveau = profil_dict.get("niveau")
    if not niveau:
        return {
            "recommandations": [],
            "sources": [],
            "erreur": "profil_incomplet",
            "message": "Veuillez renseigner au moins votre niveau d'études.",
        }

    resultat = agent.outils.analyser_profil_ml(profil_dict)
    tracabilite = {
        "question": "Analyse de profil candidat",
        "profil": f"Niveau: {niveau}, Intérêts: {profil_dict.get('centres_interet', [])}",
        "outils": "analyser_profil_ml (ExtraTrees Classifier) + RAG ISPM",
        "resultats": f"{len(resultat.get('recommandations', []))} piste(s) générée(s) avec niveau d'incertitude modéré.",
    }

    return {
        "recommandations": resultat.get("recommandations", []),
        "sources": resultat.get("sources", []),
        "tracabilite": tracabilite,
        "incertitude_globale": resultat.get("incertitude_globale", "Modérée"),
    }


def traiter_message_agent(message: str, profil: Dict[str, Any]) -> Dict[str, Any]:
    if not message.strip():
        return {"message": "Message vide", "sources": []}
    return agent.traiter_message(message, profil)


# --- Serveur HTTP Natif Standard (Fonctionne sans installer aucun package supplémentaire) ---
class GestionnaireRequetesOrientIA(BaseHTTPRequestHandler):
    def _envoyer_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._envoyer_cors()
        self.end_headers()

    def do_GET(self):
        chemin = urlparse(self.path).path
        dossier_interface = Path("interface")

        # Service des fichiers statiques web
        if chemin in ("/", "/index.html"):
            fichier = dossier_interface / "index.html"
            type_mime = "text/html; charset=utf-8"
        elif chemin == "/styles.css":
            fichier = dossier_interface / "styles.css"
            type_mime = "text/css; charset=utf-8"
        elif chemin == "/app.js":
            fichier = dossier_interface / "app.js"
            type_mime = "application/javascript; charset=utf-8"
        elif chemin == "/api-client.js":
            fichier = dossier_interface / "api-client.js"
            type_mime = "application/javascript; charset=utf-8"
        else:
            fichier = None
            type_mime = None

        if fichier and fichier.exists():
            self.send_response(200)
            self.send_header("Content-Type", type_mime)
            self._envoyer_cors()
            self.end_headers()
            self.wfile.write(fichier.read_bytes())
            return

        # Endpoints JSON
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._envoyer_cors()
        self.end_headers()

        if chemin in ("/sante", "/health"):
            reponse = {"statut": "operationnel", "service": "ORIENT'IA", "status": "ok"}
        elif chemin == "/traces":
            reponse = {"nb_traces": len(agent.historique_traces), "traces": [asdict(t) for t in agent.historique_traces]}
        else:
            reponse = {"service": "ORIENT'IA API", "routes": ["/profil/analyser", "/api/recommandation", "/agent/message", "/api/agent/chat", "/traces", "/sante"]}

        self.wfile.write(json.dumps(reponse, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        chemin = urlparse(self.path).path
        taille = int(self.headers.get("Content-Length", 0))
        corps_brut = self.rfile.read(taille).decode("utf-8") if taille > 0 else "{}"

        try:
            payload = json.loads(corps_brut) if corps_brut else {}
        except Exception:
            payload = {}

        if chemin in ("/profil/analyser", "/api/recommandation"):
            prof_data = payload.get("profil", payload) if isinstance(payload.get("profil"), dict) else payload
            reponse = traiter_analyse_profil(prof_data)
        elif chemin in ("/agent/message", "/api/agent/chat"):
            msg = payload.get("message", "")
            prof = payload.get("profil", {})
            reponse = traiter_message_agent(msg, prof)
        else:
            self.send_response(404)
            self._envoyer_cors()
            self.end_headers()
            self.wfile.write(b'{"erreur": "Route introuvable"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._envoyer_cors()
        self.end_headers()
        self.wfile.write(json.dumps(reponse, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        print(f"[API ORIENT'IA] {args[0]} - {args[1]}")


# --- Support FastAPI conditionnel si le package est installé ---
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="ORIENT'IA API",
        description="Backend d'aide à l'orientation pédagogique combinant ML, RAG et Agent conversationnel.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ProfilPayload(BaseModel):
        niveau: Optional[str] = None
        serie_bacc: Optional[str] = "C"
        moyenne_generale: Optional[float] = 12.5
        matieres_preferees: Optional[Any] = ""
        competences: Optional[str] = ""
        centres_interet: Optional[Any] = []
        projets: Optional[str] = ""
        preferences_professionnelles: Optional[str] = ""
        environnement_travail: Optional[str] = ""

    class MessagePayload(BaseModel):
        message: str
        profil: Optional[Dict[str, Any]] = None

    @app.get("/sante")
    @app.get("/health")
    def api_sante():
        return {"statut": "operationnel", "service": "ORIENT'IA", "status": "ok"}

    @app.post("/profil/analyser")
    @app.post("/api/recommandation")
    def api_profil(payload: ProfilPayload):
        return traiter_analyse_profil(payload.model_dump())

    @app.post("/agent/message")
    @app.post("/api/agent/chat")
    def api_message(payload: MessagePayload):
        return traiter_message_agent(payload.message, payload.profil or {})

    @app.get("/traces")
    def api_traces():
        return {"nb_traces": len(agent.historique_traces), "traces": [asdict(t) for t in agent.historique_traces]}

except ImportError:
    app = None


def lancer_serveur(port: int = 8000):
    adresse = ("127.0.0.1", port)
    print(f"\n========================================================")
    print(f" Serveur ORIENT'IA démarré sur http://127.0.0.1:{port}")
    print(f" Prêt à recevoir les requêtes de l'interface web.")
    print(f"========================================================\n")
    serveur_http = HTTPServer(adresse, GestionnaireRequetesOrientIA)
    try:
        serveur_http.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur ORIENT'IA.")
        serveur_http.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    lancer_serveur(port)

