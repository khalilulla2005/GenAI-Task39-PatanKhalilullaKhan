---
title: CodeLlama Assistant Deployment
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Assignment 39 — Deploying the CodeLlama Assistant

This standalone deployment package contains the working Streamlit CodeLlama coding assistant from Assignment 38, its `requirements.txt`, and configuration for **Streamlit Community Cloud** and **Hugging Face Spaces**. The app supports code generation, explanation, debugging, and optimization.

## Deployment architecture and prerequisite

The app is the Streamlit interface; the CodeLlama model runs in an Ollama service. Cloud platforms do not automatically provide the Ollama model just because `app.py` is deployed. Before cloud inference works, provide an Ollama-compatible HTTPS endpoint that is reachable from the cloud app, has `codellama:7b` installed, and is protected by a gateway that enforces authentication. Configure the endpoint and any gateway token as platform secrets, never in source control. The optional `OLLAMA_API_KEY` is sent as a Bearer token.

For local development, install Ollama and run:

```powershell
ollama pull codellama:7b
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

By default, the app connects to `http://127.0.0.1:11434`. It does not execute submitted or generated code.

## Task 1 — Deploy on Streamlit Community Cloud

1. Push this project to GitHub. The deployment entry point is `app.py`; its dependencies are declared in `requirements.txt`.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io), choose **Create app**, and select this GitHub repository, branch `main`, and `app.py`.
3. Before testing model requests, open the app's settings and add secrets for your secured Ollama gateway:

   ```toml
   OLLAMA_BASE_URL = "https://your-authenticated-ollama-gateway.example"
   OLLAMA_MODEL = "codellama:7b"
   OLLAMA_API_KEY = "your-gateway-token"
   ```

4. Deploy or reboot the app. In the deployed sidebar, select **Check Ollama connection**. Confirm that the selected model is available, then submit a short prompt such as “Write a Python function that checks whether an integer is prime.”
5. Record the assigned Streamlit URL after completing those checks.

Do not expose an unauthenticated Ollama server to the public internet. The gateway must validate the Bearer token and restrict access appropriately.

## Task 2 — Deploy on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) and select **Streamlit** as the SDK.
2. Use the Space's Git URL to publish this folder's files (or upload the project files). The YAML metadata at the top of this README identifies `app.py` as the entry point; `requirements.txt` lists the Python dependencies.
3. In the Space's **Settings**, add `OLLAMA_BASE_URL` and `OLLAMA_MODEL` as variables, and add `OLLAMA_API_KEY` as a secret if your gateway requires it.
4. Wait for the Space build to complete. Open the Space, check the Ollama connection, and test the same short prompt. Record the Space URL only after those checks pass.

Spaces host this Streamlit interface; they do not automatically start the Ollama service or download the model for this app. A configured, reachable model endpoint is required for inference.

## Task 3 — Compare deployment platforms

| Platform | Advantages | Trade-offs | Good choice when |
| --- | --- | --- | --- |
| Streamlit Community Cloud | Direct GitHub-to-app workflow, straightforward Streamlit configuration, app secrets support | App resources are managed; model hosting remains a separate requirement | You want to publish a simple Streamlit demo quickly |
| Hugging Face Spaces | Git-backed revisions, ML-focused community, multiple Space hardware options | The Ollama model still needs a separately reachable service; Space build and access settings need configuration | You want an ML demo associated with Hugging Face assets and workflows |

Both platforms can host this same interface. For either one, keep model access behind an authenticated HTTPS gateway, store credentials only in platform secrets, and test both the health check and a real model response from the deployed URL.

## Deployment verification status

**Cloud deployment and live URLs are pending.** This project is prepared for both platforms, but no Streamlit Cloud or Hugging Face Space was created from this environment: account access, deployment credentials, and a reachable Ollama model endpoint are not configured here. Do not treat the repository URL as a live app URL. After deploying with your accounts, replace this note with the verified Streamlit URL, Space URL, and test results.

## Offline validation

Run the local tests without a model download or cloud account:

```powershell
python -m unittest discover -s tests -v
```

The tests use a fake chat model and cover prompt modes, configuration, request validation, model discovery, and gateway headers. They do not prove cloud deployment or live inference.

## Project structure

```text
.
├── .streamlit/
│   └── config.toml
├── src/
│   ├── assistant.py
│   ├── config.py
│   └── prompts.py
├── tests/
│   └── test_assistant.py
├── .env.example
├── app.py
├── main.py
├── README.md
└── requirements.txt
```
