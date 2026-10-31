# Agent Playbook — développement

L’API expose `POST /v1/chat/completions`, ainsi que les événements d’outils dans `delta.agent_event`. Les fichiers `tests/test_sse_events.py` vérifient le contrat de streaming.

## Installation depuis les sources

Clonez `agent-playbook-ia` et `rag-foundry` côte à côte. Dans le dossier de l’API :

```bash
python -m venv .venv
# activez l’environnement
pip install -r requirements.txt
```

Copiez `.env.example` vers `.env` et chargez vos variables dans le shell avant le lancement. Configurez votre index documentaire, vos tables, votre workspace et vos secrets.

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Swagger est servi par FastAPI à `/docs`. Les anciennes roues compilées et données d’évaluation ont été remplacées par les packages sources et doivent être validées avec vos intégrations.

## Export OpenAPI sans backend cloud

```bash
pip install fastapi
python export_openapi.py
```

Cet export utilise des substituts pour les backends RAG ; il valide le schéma HTTP, pas une connexion au cloud.
