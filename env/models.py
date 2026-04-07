from pydantic import BaseModel

class Observation(BaseModel):
    step: int
    concentration: float
    metabolism_rate: float
    last_dose: float
    toxicity_flag: bool

class Action(BaseModel):
    dose: float


    