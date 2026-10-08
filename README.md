# H.E.L.E.N — TCN vs LSTM experiment dashboard

Interactive Streamlit demonstration of two completed local experiments: UCI Auslan glove-sequence classification and reproduction of Ricardo Colindres's Kaggle climate forecasting notebook.

## Run locally

Use Python 3.12. Install `requirements.txt`, then run `streamlit run app.py`.

The dashboard displays measured, bundled outputs. It does not retrain models, perform live webcam recognition or implement ISL-to-speech. No login or API keys are required to view it.

## Findings

Auslan: LSTM 49.69% test accuracy, TCN 30.19%, with one held-out test signer. Weather: lowest local RMSE 1.849 °C for temperature-only LSTM and 1.855 °C for two-feature TCN. These runs do not establish that TCN outperforms LSTM for ISL. The app explains why TCN remains a research candidate and what a fair ISL comparison requires.

## Free Streamlit Community Cloud deployment

Push this directory as a repository. At https://share.streamlit.io select the repository, branch `main`, entrypoint `app.py`, and Python 3.12 in advanced settings. No secrets or paid services are required. Public deployments expose the bundled public experiment results and source files.

## Reproduction

`experiments/` contains the training and explanation scripts, original requirements and documentation. The browser-upload deployment preserves these in `training_sources.zip`; extract it to access `experiments/` and `reference/`. Their paths refer to the original separate project layouts, so recreate those directories and follow their READMEs when retraining. Obtain public datasets from the cited providers. Model checkpoints and dataset archives are not shipped in the dashboard repository. The climate source notebook is preserved under `reference/`; see `NOTICE.md` and `LICENSE-2.0.txt`.

For browser uploads, `demo_assets.zip` bundles the exact same result assets and figures. The app validates its file paths and extracts the archive on first startup. No dataset download or training happens in the hosted app.

The climate reproduction preserves notebook preprocessing and recursive-forecast limitations for fidelity. They are explained in the dashboard and should be corrected for a new controlled benchmark. `assets/provenance.json` records source paths and SHA-256 hashes of bundled result assets.
