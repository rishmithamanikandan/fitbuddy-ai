import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="FitBuddy",
    description="AI Fitness Plan Generator using Google Gemini",
    version="1.0.0"
)

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None


class FitnessRequest(BaseModel):
    age: int
    goal: str
    intensity: str
    preferences: Optional[str] = ""


class FitnessResponse(BaseModel):
    plan: str


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "FitBuddy"}


@app.post("/generate-plan", response_model=FitnessResponse)
async def generate_plan(request: FitnessRequest):

    if request.age < 13 or request.age > 100:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid age."
        )

    if request.goal not in [
        "general wellness",
        "muscle gain",
        "weight management"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid fitness goal."
        )

    if request.intensity not in [
        "low",
        "moderate",
        "high"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid intensity."
        )

    if not client:
        return FitnessResponse(
            plan="Gemini API key is not configured yet. "
                 "Please add GEMINI_API_KEY to the environment."
        )

    prompt = f"""
You are FitBuddy, a safe general wellness assistant.

Create a simple 7-day fitness and wellness plan.

User age: {request.age}
Goal: {request.goal}
Preferred intensity: {request.intensity}
Preferences: {request.preferences}

Important safety rules:
- Keep the plan suitable for the user's age.
- Do not provide dangerous, extreme, or excessive exercise.
- Do not recommend starvation, crash diets, or extreme calorie restriction.
- Do not diagnose medical conditions.
- Include rest and recovery.
- Suggest stopping exercise if the user feels pain, dizziness,
  faintness, or unusual discomfort.
- For medical conditions or injuries, recommend consulting a
  qualified healthcare professional.
- Focus on healthy habits, hydration, sleep, balanced nutrition,
  and gradual activity.

Format the answer as:

Day 1:
Workout:
Wellness tip:

Day 2:
Workout:
Wellness tip:

Continue through Day 7.

At the end include:
Nutrition & Recovery Tips:
Safety Note:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        plan = response.text

        if not plan:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response."
            )

        return FitnessResponse(plan=plan)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate the plan: {str(error)}"
        )
