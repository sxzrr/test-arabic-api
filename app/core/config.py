import os

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "model/model.h5")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "utils/vectorizer.pkl")
    EXPECTED_FEATURES = int(os.getenv("EXPECTED_FEATURES", 16263))

settings = Settings()
