# models.py
# These are the data contracts for our Supply Chain Disruption Triage environment.
# They define exactly what data flows between the AI agent and the environment.

from typing import List, Optional, Dict
from pydantic import BaseModel


# ─────────────────────────────────────────────
# BASE CLASSES
# Every Action, Observation, State inherits from these
# ─────────────────────────────────────────────

class Action(BaseModel):
    """Base class for all actions"""
    pass

class Observation(BaseModel):
    """Base class for all observations"""
    done: bool = False
    reward: Optional[float] = None

class State(BaseModel):
    """Base class for all states"""
    episode_id: Optional[str] = None
    step_count: int = 0


# ─────────────────────────────────────────────
# SUPPLY CHAIN SPECIFIC MODELS
# ─────────────────────────────────────────────

class SupplierInfo(BaseModel):
    """
    Represents one supplier in the supply chain.
    Think of this as a single row in a supplier database.
    """
    supplier_id: str           # e.g. "SUP-001"
    name: str                  # e.g. "FastParts Co."
    disruption_level: str      # "none", "minor", "major", "critical"
    delay_days: int            # how many days delayed (0 if no disruption)
    daily_cost_usd: int        # cost per day to keep waiting
    products_supplied: List[str]  # which products this supplier provides


class ProductInfo(BaseModel):
    """
    Represents one product affected by disruptions.
    """
    product_id: str            # e.g. "PROD-001"
    name: str                  # e.g. "Laptop Model X"
    revenue_per_unit: int      # how much money per unit sold
    units_at_risk: int         # how many units could be lost
    days_until_stockout: int   # how many days before we run out
    affected_by_suppliers: List[str]  # which suppliers feed this product


class SupplyChainAction(Action):
    """
    What the AI agent sends to the environment each step.
    The agent looks at the situation and decides what to do
    about ONE supplier at a time.
    """
    supplier_id: str           # which supplier this decision is about
    decision: str              # "wait", "find_alternate", "use_safety_stock", "expedite"
    reasoning: str             # why the agent made this choice (for logging)


class SupplyChainObservation(Observation):
    """
    What the environment sends back to the AI agent after each action.
    This is everything the agent needs to make its next decision.
    
    Inherits from Observation, so it already has:
    - done: bool
    - reward: Optional[float]
    """
    # Current situation
    disrupted_suppliers: List[SupplierInfo]    # suppliers with problems
    affected_products: List[ProductInfo]        # products at risk
    
    # Budget tracking
    total_budget_usd: int                       # total money available
    budget_remaining_usd: int                   # money left to spend
    
    # Progress
    decisions_made: int                         # how many suppliers handled
    decisions_remaining: int                    # how many left to handle
    
    # Feedback
    last_action_result: str                     # what happened after last action
    current_score: float                        # running score 0.0 to 1.0
    message: str                                # human readable status message


class SupplyChainState(State):
    """
    Behind-the-scenes tracking of the episode.
    Not shown to the agent directly — used for scoring and logging.
    
    Inherits from State, so it already has:
    - episode_id: Optional[str]
    - step_count: int
    """
    task_id: str = ""               # "task1_easy", "task2_medium", "task3_hard"
    total_budget_usd: int = 0       # starting budget for this episode
    budget_remaining_usd: int = 0   # current remaining budget
    correct_decisions: int = 0      # how many decisions were optimal
    total_decisions: int = 0        # total decisions needed
    critical_errors: int = 0        # dangerous mistakes made
    final_score: float = 0.0        # end of episode score