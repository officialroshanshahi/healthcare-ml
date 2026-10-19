from enum import Enum
from typing import List
from pydantic import BaseModel, Field

# enum mapped exactly to the values in healthcare_dataset.csv
class TestResultType(str, Enum):
    NORMAL = "Normal"
    ABNORMAL = "Abnormal"
    INCONCLUSIVE = "Inconclusive"

class PatientData(BaseModel):
    Name: str = Field(..., example="Bobby JacksOn")
    Age: int = Field(..., example=30)
    Gender: str = Field(..., example="Male")
    Blood_Type: str = Field(..., alias="Blood Type", example="B-")
    Medical_Condition: str = Field(..., alias="Medical Condition", example="Cancer")
    Date_of_Admission: str = Field(..., alias="Date of Admission", example="2024-01-31")
    Doctor: str = Field(..., example="Matthew Smith")
    Hospital: str = Field(..., example="Sons and Miller")
    Insurance_Provider: str = Field(..., alias="Insurance Provider", example="Blue Cross")
    Billing_Amount: float = Field(..., alias="Billing Amount", example=18856.28)
    Room_Number: int = Field(..., alias="Room Number", example=328)
    Admission_Type: str = Field(..., alias="Admission Type", example="Urgent")
    Discharge_Date: str = Field(..., alias="Discharge Date", example="2024-02-02")
    Medication: str = Field(..., example="Paracetamol")
    class Config: populate_by_name = True

# other schemas
class UserAuth(BaseModel): username: str; password: str
class ReqSchema(BaseModel): data: PatientData
class Batch(BaseModel): data: List[PatientData]
class Label(BaseModel): request_id: str; test_results: int