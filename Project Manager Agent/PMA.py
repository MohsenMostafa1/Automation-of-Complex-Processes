# Simplifying AI Agent Quality Checklist for PM

import os
import sqlite3
from enum import Enum
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import pipeline
import requests
from dotenv import load_dotenv
import logging

# --- Configuration ---
load_dotenv()
app = FastAPI(title="Open-Source Project Manager Agent")

# Database
DB_NAME = "sprints.db"
conn = sqlite3.connect(DB_NAME)
conn.execute("""
CREATE TABLE IF NOT EXISTS sprints (
    id INTEGER PRIMARY KEY,
    goal TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT,
    assignee_name TEXT,
    status TEXT,
    sprint_id INTEGER,
    taiga_id INTEGER,
    FOREIGN KEY(sprint_id) REFERENCES sprints(id)
)
""")
conn.commit()

# Taiga
TAIGA_API_URL = os.getenv("TAIGA_API_URL", "http://localhost:8000/api/v1")
TAIGA_AUTH = (os.getenv("TAIGA_USER"), os.getenv("TAIGA_PASSWORD"))

# Hugging Face
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# --- Models ---
class TaskStatus(str, Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"

class TeamMember(BaseModel):
    name: str
    role: str  # "Dev", "QA", "Designer"
    email: Optional[str] = None

class TaskCreate(BaseModel):
    title: str
    assignee: Optional[TeamMember] = None
    status: TaskStatus = TaskStatus.TODO

class SprintCreate(BaseModel):
    goal: str
    project_id: int  # Taiga project ID
    team: List[TeamMember]

# --- Services ---
class LLMService:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model=HF_MODEL,
            device_map="auto"  # Uses GPU if available
        )
        logging.info(f"Loaded LLM: {HF_MODEL}")

    def generate_tasks(self, goal: str) -> List[str]:
        prompt = f"""Break this software development goal into subtasks:
        Goal: {goal}
        Tasks:"""
        output = self.pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )
        return [
            line.strip() 
            for line in output[0]["generated_text"].split("\n") 
            if line.startswith("-")
        ][:5]  # Limit to 5 tasks

class TaigaService:
    def __init__(self):
        self.auth_token = self._authenticate()

    def _authenticate(self) -> str:
        response = requests.post(
            f"{TAIGA_API_URL}/auth",
            json={"username": TAIGA_AUTH[0], "password": TAIGA_AUTH[1], "type": "normal"}
        )
        response.raise_for_status()
        return response.json()["auth_token"]

    def create_task(self, title: str, project_id: int, status: str) -> int:
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = requests.post(
            f"{TAIGA_API_URL}/tasks",
            headers=headers,
            json={
                "subject": title,
                "project": project_id,
                "status": status
            }
        )
        response.raise_for_status()
        return response.json()["id"]

# --- Core Agent ---
class ProjectManagerAgent:
    def __init__(self):
        self.llm = LLMService()
        self.taiga = TaigaService()

    def _assign_task(self, task_title: str, team: List[TeamMember]) -> Optional[TeamMember]:
        role_keywords = {
            "Dev": ["implement", "build", "refactor", "write"],
            "QA": ["test", "validate", "verify"],
            "Designer": ["design", "ui", "ux", "research"]
        }
        for member in team:
            keywords = role_keywords.get(member.role, [])
            if any(kw in task_title.lower() for kw in keywords):
                return member
        return None

    def plan_sprint(self, goal: str, project_id: int, team: List[TeamMember]):
        # Generate tasks with LLM
        task_titles = self.llm.generate_tasks(goal)
        
        # Create sprint in DB
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sprints (goal) VALUES (?) RETURNING id",
            (goal,)
        )
        sprint_id = cursor.fetchone()[0]
        
        # Process tasks
        tasks = []
        for i, title in enumerate(task_titles):
            # Assign task
            assignee = self._assign_task(title, team)
            
            # Create in Taiga
            status = TaskStatus.IN_PROGRESS if assignee else TaskStatus.TODO
            try:
                taiga_id = self.taiga.create_task(title, project_id, status.value)
            except Exception as e:
                logging.error(f"Taiga error: {str(e)}")
                taiga_id = None
            
            # Save to DB
            task_id = f"T{sprint_id}-{i}"
            cursor.execute(
                """INSERT INTO tasks 
                (id, title, assignee_name, status, sprint_id, taiga_id) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, title, assignee.name if assignee else None, 
                 status.value, sprint_id, taiga_id)
            )
            tasks.append({
                "id": task_id,
                "title": title,
                "assignee": assignee.dict() if assignee else None,
                "status": status,
                "taiga_id": taiga_id
            })
        
        conn.commit()
        return {"sprint_id": sprint_id, "tasks": tasks}

# --- API Endpoints ---
manager = ProjectManagerAgent()

@app.post("/sprints/", status_code=201)
def create_sprint(sprint: SprintCreate):
    try:
        result = manager.plan_sprint(
            goal=sprint.goal,
            project_id=sprint.project_id,
            team=sprint.team
        )
        return result
    except Exception as e:
        logging.exception("Sprint planning failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sprints/{sprint_id}")
def get_sprint(sprint_id: int):
    cursor = conn.cursor()
    cursor.execute("SELECT goal FROM sprints WHERE id = ?", (sprint_id,))
    sprint = cursor.fetchone()
    if not sprint:
        raise HTTPException(status_code=404, detail="Sprint not found")
    
    cursor.execute(
        """SELECT id, title, assignee_name, status, taiga_id 
        FROM tasks WHERE sprint_id = ?""", 
        (sprint_id,)
    )
    tasks = [
        {
            "id": row[0],
            "title": row[1],
            "assignee": row[2],
            "status": row[3],
            "taiga_id": row[4]
        }
        for row in cursor.fetchall()
    ]
    return {"goal": sprint[0], "tasks": tasks}

# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
