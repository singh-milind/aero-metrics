import json
import os
from src.api.schemas.predictor.reasoning import PredictorReasoning
from dotenv import load_dotenv
from google import genai

from src.api.services.predictor_metadata import (
    get_feature_metadata,
    get_system_prompt
)

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-3.5-flash"


def generate_reasoning(shap_result: dict) -> str:

    metadata = get_feature_metadata()
    system_prompt = get_system_prompt()

    prompt = f"""
        Explain the following local air-quality model prediction.

        SHAP RESULT:
        {json.dumps(shap_result, indent=2)}

        FEATURE METADATA:
        {json.dumps(metadata, indent=2)}

        Follow the system instructions exactly.
        """

    response = client.models.generate_content(
            model=MODEL_NAME,
            contents=system_prompt + "\n\n" + prompt,
        )

    return response.text.strip()


    