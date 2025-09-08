from pydantic import BaseModel
from typing import List

class PredictResponse(BaseModel):
    probabilities: List[float]  # P(fake)
    labels: List[int]           # 1=fake, 0=real
