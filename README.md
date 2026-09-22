# FINCORP Customer Message Intelligence

A Streamlit deployment companion for the **FINCORP CRISP-DM Sentiment and Emotion** notebook. The application accepts one customer message and produces two independent predictions:

- sentiment: negative, neutral, or positive;
- emotion: anger, disgust, fear, happiness, other, sadness, or surprise.

It also recommends human review whenever either model's confidence is below the threshold recorded in the deployment metadata.

## Repository structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── models/
    ├── fincorp_sentiment_pipeline.pkl
    ├── fincorp_emotion_pipeline.pkl
    ├── sentiment_emotion_metadata.json
    └── README.md
```

The ZIP deliberately does not contain trained model binaries. Run the Colab notebook first and download these three files from:

`My Drive/Fintech/Text_Models`

Then place them in the repository's `models/` folder.

## Local setup

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and change the access code.
4. Add the three model artifacts to `models/`.
5. Run:

   ```bash
   streamlit run streamlit_app.py
   ```

## Streamlit Community Cloud

1. Upload the package contents to a GitHub repository.
2. Ensure the three deployment artifacts are committed inside `models/`.
3. Create a Streamlit app using `streamlit_app.py` as the entry point.
4. Under **App settings → Secrets**, add:

   ```toml
   APP_ACCESS_CODE = "replace-with-your-code"
   ```

5. Deploy the application.

## Data and privacy

The application contains no FINCORP raw dataset and provides no raw-data download. It processes only the message entered by the current user. Avoid entering confidential or personally identifiable information.

## Intended use

This is an educational decision-support demonstration. Model outputs can support triage and aggregate monitoring but should not replace human judgement in high-impact cases.

