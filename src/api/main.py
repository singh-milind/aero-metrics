from fastapi import FastAPI
from src.api.routes.predictor import router as predictor_router
from src.api.routes.forecaster_auto import router as forecaster_router
from src.api.routes.forecaster_manual import router as forecaster_manual_router 
from src.api.routes.predictor_explainer import router as predictor_explainer_router
from src.api.routes.forecaster_explainer import router as forecaster_explainer_router


app = FastAPI(title="Aero Metrics API")

app.include_router(predictor_router, prefix="/api")
app.include_router(forecaster_router,prefix="/api")
app.include_router(forecaster_manual_router, prefix="/api")

app.include_router(predictor_explainer_router,prefix="/api/explainer")
app.include_router(forecaster_explainer_router,prefix="/api/explainer")

@app.get("/")
def root():
    return {"message": "API is running"}