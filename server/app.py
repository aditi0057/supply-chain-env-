# server/app.py
# This file turns our environment into a running web server.
# FastAPI handles all the HTTP endpoints automatically.

import sys
import os

# Add parent folder to path so we can import models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from environment import SupplyChainEnvironment
from models import SupplyChainAction

# ── Create the FastAPI app ──
app = FastAPI(
    title="Supply Chain Disruption Triage Environment",
    description="An OpenEnv environment where AI agents learn to triage "
                "supply chain disruptions by making cost-effective decisions "
                "under budget constraints.",
    version="1.0.0",
)

# ── Allow connections from anywhere ──
# This is needed so HuggingFace Spaces can talk to your server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Create one shared environment instance ──
env = SupplyChainEnvironment()


# ── Request body models ──
# These define what JSON the server expects to receive

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1_easy"
    episode_id: Optional[str] = None

class StepRequest(BaseModel):
    supplier_id: str
    decision: str
    reasoning: str = ""


# ────────────────────────────────────────────
# ENDPOINTS
# ────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if server is running"""
    return {"status": "healthy", "environment": "supply-chain-triage"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Send task_id to choose difficulty:
    - "task1_easy"
    - "task2_medium"  
    - "task3_hard"
    """
    observation = env.reset(
        task_id=request.task_id,
        episode_id=request.episode_id,
    )
    return {
        "observation": observation.model_dump(),
        "done": observation.done,
        "reward": observation.reward,
        "message": observation.message,
    }


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the environment.
    Send supplier_id and decision.
    
    Valid decisions:
    - "wait"
    - "find_alternate"
    - "use_safety_stock"
    - "expedite"
    """
    action = SupplyChainAction(
        supplier_id=request.supplier_id,
        decision=request.decision,
        reasoning=request.reasoning,
    )
    observation = env.step(action)
    return {
        "observation": observation.model_dump(),
        "done": observation.done,
        "reward": observation.reward,
        "message": observation.message,
    }


@app.get("/state")
def state():
    """Get current episode state"""
    return env.state.model_dump()


@app.get("/tasks")
def list_tasks():
    """
    List all available tasks with descriptions.
    Useful for agents to know what tasks exist.
    """
    return {
        "tasks": [
            {
                "task_id": "task1_easy",
                "difficulty": "easy",
                "description": "3 suppliers disrupted, clear signals, $30,000 budget",
                "expected_score_range": "0.7 - 1.0",
            },
            {
                "task_id": "task2_medium",
                "difficulty": "medium",
                "description": "6 suppliers disrupted, budget tradeoffs, $50,000 budget",
                "expected_score_range": "0.5 - 0.8",
            },
            {
                "task_id": "task3_hard",
                "difficulty": "hard",
                "description": "10 suppliers disrupted, complex dependencies, $70,000 budget",
                "expected_score_range": "0.3 - 0.7",
            },
        ]
    }


@app.get("/actions")
def list_actions():
    """
    List all valid decisions with costs and when to use them.
    Helps agents understand their action space.
    """
    return {
        "valid_decisions": [
            {
                "decision": "wait",
                "cost_usd": 0,
                "best_for": "minor disruptions",
                "description": "Wait for supplier to recover naturally",
            },
            {
                "decision": "find_alternate",
                "cost_usd": 8000,
                "best_for": "major disruptions",
                "description": "Source from a different supplier",
            },
            {
                "decision": "use_safety_stock",
                "cost_usd": 5000,
                "best_for": "short term coverage",
                "description": "Use emergency inventory reserves",
            },
            {
                "decision": "expedite",
                "cost_usd": 15000,
                "best_for": "critical disruptions",
                "description": "Rush order via premium shipping",
            },
        ]
    }