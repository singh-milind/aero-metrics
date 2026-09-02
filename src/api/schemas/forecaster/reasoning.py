from pydantic import BaseModel
from typing import List,Annotated,Union


class Data(BaseModel):
    feature: Annotated[str, "The feature name"]
    value: Annotated[Union[float, str], "The feature value"]
    shap_value: Annotated[float, "The SHAP value for the feature"]
    abs_shap: Annotated[float, "The absolute SHAP value for the feature"]
    impact: Annotated[str, "The impact of the feature on the prediction"]



class ForecasterReasoning(BaseModel):
    target: Annotated[str, "The target variable for the prediction"]
    prediction: Annotated[float, "The predicted value for the target variable"]
    base_value: Annotated[float, "The base value for the prediction"]
    data: Annotated[List[Data], "The SHAP values for each feature"]