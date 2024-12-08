from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Text Classification API")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Welcome to the Text Classification API!"}
