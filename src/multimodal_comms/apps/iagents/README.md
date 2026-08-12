# iAgents application

iAgents is a Flask application for collaborative tasks in which agents have
different private information. It includes the web interface, task prompts,
provider backends, database initialization, and offline benchmark runner.

For reproducible communication experiments, prefer the database-independent
offline path in `experiments/iagents/README.md`. To use the web application,
configure `config/global.yaml`, initialize MySQL with:

```bash
python -m multimodal_comms.apps.iagents.create_database
python -m multimodal_comms.apps.iagents.app
```

Then open the configured host and port in a browser. Full application
dependencies are listed in `requirements-app.txt`; the root Conda environment
covers the repository's headless smoke path. Store API keys, database
passwords, and Flask secrets outside tracked configuration files.

The application supports hosted providers and local Ollama-style backends.
RAG and file upload require the optional LlamaIndex dependencies and an
embedding model. These services are not started by smoke tests.
