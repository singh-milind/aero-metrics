from fastapi import FastAPI
from src.api.routes.predictor import router as predictor_router
from src.api.routes.forecaster_auto import router as forecaster_router
app = FastAPI(title="Aero Metrics API")

app.include_router(predictor_router, prefix="/api")
app.include_router(forecaster_router,prefix="/api")

@app.get("/")
def root():
    return {"message": "API is running"}