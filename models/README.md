# Required model files

After running the FINCORP Colab notebook, copy the following files from `My Drive/Fintech/Text_Models` into this folder:

1. `fincorp_sentiment_pipeline.pkl`
2. `fincorp_emotion_pipeline.pkl`
3. `sentiment_emotion_metadata.json`

The two `.pkl` files must contain fitted scikit-learn pipelines with named steps `tfidf` and `classifier`. The classifier must support `predict_proba` and expose `classes_`.

