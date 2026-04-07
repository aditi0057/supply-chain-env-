# server/environment.py
# This is the brain of the Supply Chain Disruption Triage environment.
# It simulates real supply chain crisis scenarios and scores the AI agent's decisions.

import uuid
import random
from typing import Dict, List, Optional, Tuple

# We add the parent folder to path so we can import models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
    SupplierInfo,
    ProductInfo,
)


# ─────────────────────────────────────────────────────────────
# SCENARIO DATA
# These are our pre-built crisis scenarios.
# All data is generated here in Python — no external files needed.
# ─────────────────────────────────────────────────────────────

# The optimal decision for each disruption level.
# This is what a perfect supply chain manager would do.
OPTIMAL_DECISIONS = {
    "minor":    "wait",              # small delay? just wait, cheapest option
    "major":    "find_alternate",    # serious problem? find another supplier
    "critical": "expedite",          # emergency? pay extra to rush it
}

# Cost of each decision in USD
DECISION_COSTS = {
    "wait":             0,       # free — just wait
    "find_alternate":   8000,    # costs money to onboard new supplier
    "use_safety_stock": 5000,    # costs money to use emergency inventory
    "expedite":         15000,   # most expensive — rush shipping/production
}

# How well each decision works for each disruption level
# 1.0 = perfect choice, 0.5 = okay, 0.1 = bad choice
DECISION_EFFECTIVENESS = {
    "minor": {
        "wait":             1.0,   # perfect — no need to spend money
        "use_safety_stock": 0.6,   # works but wastes money
        "find_alternate":   0.4,   # overkill for minor issue
        "expedite":         0.2,   # very wasteful for minor issue
    },
    "major": {
        "find_alternate":   1.0,   # perfect — get a new supplier
        "use_safety_stock": 0.7,   # okay short term fix
        "expedite":         0.5,   # works but expensive
        "wait":             0.1,   # bad — major issue needs action
    },
    "critical": {
        "expedite":         1.0,   # perfect — emergency needs urgent action
        "find_alternate":   0.7,   # good but slower than expedite
        "use_safety_stock": 0.4,   # buys time but doesn't fix problem
        "wait":             0.0,   # terrible — critical issue can't wait
    },
}


# ─────────────────────────────────────────────────────────────
# SCENARIO BUILDER
# Creates the three task scenarios
# ─────────────────────────────────────────────────────────────

def build_task1_easy() -> Tuple[List[SupplierInfo], List[ProductInfo], int]:
    """
    Task 1: Easy
    3 suppliers disrupted, clear signals, generous budget.
    A good agent should score 0.8+ here.
    """
    suppliers = [
        SupplierInfo(
            supplier_id="SUP-001",
            name="QuickParts Ltd",
            disruption_level="minor",
            delay_days=3,
            daily_cost_usd=500,
            products_supplied=["PROD-001"],
        ),
        SupplierInfo(
            supplier_id="SUP-002",
            name="CoreComponents Inc",
            disruption_level="major",
            delay_days=14,
            daily_cost_usd=2000,
            products_supplied=["PROD-002", "PROD-003"],
        ),
        SupplierInfo(
            supplier_id="SUP-003",
            name="RapidSupply Co",
            disruption_level="critical",
            delay_days=30,
            daily_cost_usd=5000,
            products_supplied=["PROD-004"],
        ),
    ]

    products = [
        ProductInfo(
            product_id="PROD-001",
            name="Office Chair",
            revenue_per_unit=300,
            units_at_risk=100,
            days_until_stockout=10,
            affected_by_suppliers=["SUP-001"],
        ),
        ProductInfo(
            product_id="PROD-002",
            name="Laptop Stand",
            revenue_per_unit=150,
            units_at_risk=200,
            days_until_stockout=7,
            affected_by_suppliers=["SUP-002"],
        ),
        ProductInfo(
            product_id="PROD-003",
            name="Keyboard",
            revenue_per_unit=80,
            units_at_risk=500,
            days_until_stockout=7,
            affected_by_suppliers=["SUP-002"],
        ),
        ProductInfo(
            product_id="PROD-004",
            name="Server Unit",
            revenue_per_unit=5000,
            units_at_risk=20,
            days_until_stockout=3,
            affected_by_suppliers=["SUP-003"],
        ),
    ]

    budget = 30000
    return suppliers, products, budget


def build_task2_medium() -> Tuple[List[SupplierInfo], List[ProductInfo], int]:
    """
    Task 2: Medium
    6 suppliers disrupted, some tricky tradeoffs, tighter budget.
    Budget forces the agent to prioritize — can't fix everything.
    A good agent should score 0.6-0.8 here.
    """
    suppliers = [
        SupplierInfo(
            supplier_id="SUP-001",
            name="AlphaManufacturing",
            disruption_level="minor",
            delay_days=2,
            daily_cost_usd=300,
            products_supplied=["PROD-001"],
        ),
        SupplierInfo(
            supplier_id="SUP-002",
            name="BetaComponents",
            disruption_level="major",
            delay_days=10,
            daily_cost_usd=1500,
            products_supplied=["PROD-002"],
        ),
        SupplierInfo(
            supplier_id="SUP-003",
            name="GammaParts",
            disruption_level="critical",
            delay_days=21,
            daily_cost_usd=4000,
            products_supplied=["PROD-003", "PROD-004"],
        ),
        SupplierInfo(
            supplier_id="SUP-004",
            name="DeltaSupply",
            disruption_level="major",
            delay_days=12,
            daily_cost_usd=1800,
            products_supplied=["PROD-005"],
        ),
        SupplierInfo(
            supplier_id="SUP-005",
            name="EpsilonLogistics",
            disruption_level="minor",
            delay_days=4,
            daily_cost_usd=600,
            products_supplied=["PROD-001", "PROD-002"],
        ),
        SupplierInfo(
            supplier_id="SUP-006",
            name="ZetaWholesale",
            disruption_level="critical",
            delay_days=28,
            daily_cost_usd=6000,
            products_supplied=["PROD-006"],
        ),
    ]

    products = [
        ProductInfo(
            product_id="PROD-001",
            name="Smartphone Case",
            revenue_per_unit=25,
            units_at_risk=1000,
            days_until_stockout=8,
            affected_by_suppliers=["SUP-001", "SUP-005"],
        ),
        ProductInfo(
            product_id="PROD-002",
            name="Wireless Charger",
            revenue_per_unit=45,
            units_at_risk=600,
            days_until_stockout=6,
            affected_by_suppliers=["SUP-002", "SUP-005"],
        ),
        ProductInfo(
            product_id="PROD-003",
            name="Gaming Monitor",
            revenue_per_unit=400,
            units_at_risk=150,
            days_until_stockout=4,
            affected_by_suppliers=["SUP-003"],
        ),
        ProductInfo(
            product_id="PROD-004",
            name="Graphics Card",
            revenue_per_unit=800,
            units_at_risk=80,
            days_until_stockout=4,
            affected_by_suppliers=["SUP-003"],
        ),
        ProductInfo(
            product_id="PROD-005",
            name="SSD Drive",
            revenue_per_unit=120,
            units_at_risk=400,
            days_until_stockout=5,
            affected_by_suppliers=["SUP-004"],
        ),
        ProductInfo(
            product_id="PROD-006",
            name="Industrial Robot Arm",
            revenue_per_unit=15000,
            units_at_risk=5,
            days_until_stockout=2,
            affected_by_suppliers=["SUP-006"],
        ),
    ]

    budget = 50000
    return suppliers, products, budget


def build_task3_hard() -> Tuple[List[SupplierInfo], List[ProductInfo], int]:
    """
    Task 3: Hard
    10 suppliers disrupted, complex dependencies, tight budget.
    Budget is NOT enough to fix everything — agent must prioritize
    by revenue impact and urgency. Critical thinking required.
    A good agent should score 0.5-0.7 here. Frontier models may hit 0.8.
    """
    suppliers = [
        SupplierInfo(
            supplier_id="SUP-001",
            name="NorthStar Materials",
            disruption_level="minor",
            delay_days=2,
            daily_cost_usd=200,
            products_supplied=["PROD-001"],
        ),
        SupplierInfo(
            supplier_id="SUP-002",
            name="SouthBridge Parts",
            disruption_level="major",
            delay_days=8,
            daily_cost_usd=1200,
            products_supplied=["PROD-002", "PROD-003"],
        ),
        SupplierInfo(
            supplier_id="SUP-003",
            name="EastWind Components",
            disruption_level="critical",
            delay_days=25,
            daily_cost_usd=5000,
            products_supplied=["PROD-004"],
        ),
        SupplierInfo(
            supplier_id="SUP-004",
            name="WestCoast Supply",
            disruption_level="major",
            delay_days=11,
            daily_cost_usd=1600,
            products_supplied=["PROD-005", "PROD-006"],
        ),
        SupplierInfo(
            supplier_id="SUP-005",
            name="CentralHub Logistics",
            disruption_level="critical",
            delay_days=30,
            daily_cost_usd=7000,
            products_supplied=["PROD-007"],
        ),
        SupplierInfo(
            supplier_id="SUP-006",
            name="MidTown Manufacturing",
            disruption_level="minor",
            delay_days=3,
            daily_cost_usd=400,
            products_supplied=["PROD-008"],
        ),
        SupplierInfo(
            supplier_id="SUP-007",
            name="HighTech Imports",
            disruption_level="major",
            delay_days=14,
            daily_cost_usd=2200,
            products_supplied=["PROD-004", "PROD-009"],
        ),
        SupplierInfo(
            supplier_id="SUP-008",
            name="GlobalParts Network",
            disruption_level="critical",
            delay_days=20,
            daily_cost_usd=4500,
            products_supplied=["PROD-010"],
        ),
        SupplierInfo(
            supplier_id="SUP-009",
            name="FastTrack Wholesale",
            disruption_level="minor",
            delay_days=5,
            daily_cost_usd=700,
            products_supplied=["PROD-001", "PROD-002"],
        ),
        SupplierInfo(
            supplier_id="SUP-010",
            name="PrimeLine Distributors",
            disruption_level="major",
            delay_days=9,
            daily_cost_usd=1900,
            products_supplied=["PROD-006", "PROD-010"],
        ),
    ]

    products = [
        ProductInfo(
            product_id="PROD-001",
            name="USB Cable",
            revenue_per_unit=10,
            units_at_risk=2000,
            days_until_stockout=12,
            affected_by_suppliers=["SUP-001", "SUP-009"],
        ),
        ProductInfo(
            product_id="PROD-002",
            name="Bluetooth Speaker",
            revenue_per_unit=60,
            units_at_risk=500,
            days_until_stockout=7,
            affected_by_suppliers=["SUP-002", "SUP-009"],
        ),
        ProductInfo(
            product_id="PROD-003",
            name="Webcam",
            revenue_per_unit=90,
            units_at_risk=300,
            days_until_stockout=6,
            affected_by_suppliers=["SUP-002"],
        ),
        ProductInfo(
            product_id="PROD-004",
            name="AI Accelerator Chip",
            revenue_per_unit=2000,
            units_at_risk=100,
            days_until_stockout=3,
            affected_by_suppliers=["SUP-003", "SUP-007"],
        ),
        ProductInfo(
            product_id="PROD-005",
            name="Electric Motor",
            revenue_per_unit=350,
            units_at_risk=200,
            days_until_stockout=5,
            affected_by_suppliers=["SUP-004"],
        ),
        ProductInfo(
            product_id="PROD-006",
            name="Power Supply Unit",
            revenue_per_unit=120,
            units_at_risk=400,
            days_until_stockout=5,
            affected_by_suppliers=["SUP-004", "SUP-010"],
        ),
        ProductInfo(
            product_id="PROD-007",
            name="Medical Sensor Array",
            revenue_per_unit=8000,
            units_at_risk=30,
            days_until_stockout=2,
            affected_by_suppliers=["SUP-005"],
        ),
        ProductInfo(
            product_id="PROD-008",
            name="Circuit Board",
            revenue_per_unit=180,
            units_at_risk=600,
            days_until_stockout=9,
            affected_by_suppliers=["SUP-006"],
        ),
        ProductInfo(
            product_id="PROD-009",
            name="Radar Module",
            revenue_per_unit=3500,
            units_at_risk=40,
            days_until_stockout=4,
            affected_by_suppliers=["SUP-007"],
        ),
        ProductInfo(
            product_id="PROD-010",
            name="Quantum Memory Unit",
            revenue_per_unit=12000,
            units_at_risk=15,
            days_until_stockout=3,
            affected_by_suppliers=["SUP-008", "SUP-010"],
        ),
    ]

    budget = 70000
    return suppliers, products, budget


# ─────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────

class SupplyChainEnvironment:
    """
    The main environment class.
    Implements reset(), step(), and state property.
    """

    SUPPORTED_TASKS = ["task1_easy", "task2_medium", "task3_hard"]

    def __init__(self):
        # These will be set properly when reset() is called
        self._state = SupplyChainState()
        self._suppliers: List[SupplierInfo] = []
        self._products: List[ProductInfo] = []
        self._pending_suppliers: List[SupplierInfo] = []
        self._decisions_log: Dict[str, str] = {}
        self._total_budget: int = 0
        self._budget_remaining: int = 0
        self._correct_decisions: int = 0
        self._total_decisions: int = 0
        self._critical_errors: int = 0
        self._current_score: float = 0.0

    def reset(self, task_id: str = "task1_easy",
              episode_id: Optional[str] = None) -> SupplyChainObservation:
        """
        Start a fresh episode.
        task_id controls which scenario to load.
        """
        # Validate task
        if task_id not in self.SUPPORTED_TASKS:
            task_id = "task1_easy"

        # Load the right scenario
        if task_id == "task1_easy":
            suppliers, products, budget = build_task1_easy()
        elif task_id == "task2_medium":
            suppliers, products, budget = build_task2_medium()
        else:
            suppliers, products, budget = build_task3_hard()

        # Set up fresh episode state
        self._suppliers = suppliers
        self._products = products
        self._pending_suppliers = list(suppliers)  # copy — we'll pop from this
        self._decisions_log = {}
        self._total_budget = budget
        self._budget_remaining = budget
        self._correct_decisions = 0
        self._total_decisions = len(suppliers)
        self._critical_errors = 0
        self._current_score = 0.0

        # Set up state tracking
        self._state = SupplyChainState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            total_budget_usd=budget,
            budget_remaining_usd=budget,
            correct_decisions=0,
            total_decisions=len(suppliers),
            critical_errors=0,
            final_score=0.0,
        )

        return SupplyChainObservation(
            done=False,
            reward=None,
            disrupted_suppliers=self._pending_suppliers.copy(),
            affected_products=self._products,
            total_budget_usd=self._total_budget,
            budget_remaining_usd=self._budget_remaining,
            decisions_made=0,
            decisions_remaining=len(self._pending_suppliers),
            last_action_result="Episode started. Assess disruptions and make decisions.",
            current_score=0.0,
            message=f"Task: {task_id} | {len(suppliers)} suppliers disrupted | "
                    f"Budget: ${budget:,} | Make triage decisions for each supplier.",
        )

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """
        Process one agent decision.
        The agent picks one supplier and decides what to do with it.
        """
        self._state.step_count += 1

        # ── Find the supplier the agent is deciding about ──
        supplier = self._find_supplier(action.supplier_id)

        if supplier is None:
            # Agent gave an invalid supplier ID — penalize
            return self._make_observation(
                reward=-0.1,
                done=False,
                result=f"Unknown supplier '{action.supplier_id}'. "
                       f"Valid IDs: {[s.supplier_id for s in self._pending_suppliers]}",
            )

        if supplier not in self._pending_suppliers:
            # Agent is trying to decide about a supplier already handled
            return self._make_observation(
                reward=-0.05,
                done=False,
                result=f"Supplier '{action.supplier_id}' already handled. "
                       f"Remaining: {[s.supplier_id for s in self._pending_suppliers]}",
            )

        # ── Score the decision ──
        reward, result_message = self._score_decision(supplier, action.decision)

        # ── Apply cost ──
        cost = DECISION_COSTS.get(action.decision, 0)
        if cost > self._budget_remaining:
            # Over budget — penalize and don't apply decision
            reward = -0.2
            result_message = (
                f"Decision '{action.decision}' costs ${cost:,} but only "
                f"${self._budget_remaining:,} remaining. Decision rejected."
            )
        else:
            # Valid — apply the decision
            self._budget_remaining -= cost
            self._pending_suppliers.remove(supplier)
            self._decisions_log[supplier.supplier_id] = action.decision

            # Track correct decisions
            optimal = OPTIMAL_DECISIONS.get(supplier.disruption_level, "wait")
            if action.decision == optimal:
                self._correct_decisions += 1

            # Track critical errors
            # A critical error = choosing "wait" for a critical supplier
            if supplier.disruption_level == "critical" and action.decision == "wait":
                self._critical_errors += 1

        # ── Update running score ──
        self._current_score = self._calculate_score()
        self._state.budget_remaining_usd = self._budget_remaining
        self._state.correct_decisions = self._correct_decisions
        self._state.critical_errors = self._critical_errors

        # ── Check if episode is done ──
        done = len(self._pending_suppliers) == 0

        if done:
            final_score = self._calculate_final_score()
            self._state.final_score = final_score
            reward = final_score
            result_message += f" | Episode complete! Final score: {final_score:.2f}"

        return self._make_observation(
            reward=reward,
            done=done,
            result=result_message,
        )

    @property
    def state(self) -> SupplyChainState:
        """Return current episode state"""
        return self._state

    # ─────────────────────────────────────────
    # PRIVATE HELPER METHODS
    # ─────────────────────────────────────────

    def _find_supplier(self, supplier_id: str) -> Optional[SupplierInfo]:
        """Find a supplier by ID from all suppliers (not just pending)"""
        for s in self._suppliers:
            if s.supplier_id == supplier_id:
                return s
        return None

    def _score_decision(
        self, supplier: SupplierInfo, decision: str
    ) -> Tuple[float, str]:
        """
        Score how good the agent's decision was for this supplier.
        Returns (reward, message).
        """
        level = supplier.disruption_level
        effectiveness = DECISION_EFFECTIVENESS.get(level, {})
        score = effectiveness.get(decision, 0.1)
        optimal = OPTIMAL_DECISIONS.get(level, "wait")
        cost = DECISION_COSTS.get(decision, 0)

        if score == 1.0:
            msg = (
                f"Optimal decision for {supplier.name} "
                f"({level} disruption)! '{decision}' is exactly right. "
                f"Cost: ${cost:,}"
            )
        elif score >= 0.6:
            msg = (
                f"Good decision for {supplier.name}. '{decision}' works "
                f"but '{optimal}' would be optimal. Cost: ${cost:,}"
            )
        elif score >= 0.3:
            msg = (
                f"Suboptimal decision for {supplier.name}. "
                f"'{decision}' is not ideal for {level} disruption. "
                f"Consider '{optimal}' next time."
            )
        else:
            msg = (
                f"Poor decision for {supplier.name}. "
                f"'{decision}' for a {level} disruption is dangerous. "
                f"Should be '{optimal}'."
            )

        return score * 0.3, msg  # Each step reward is partial (max 0.3 per step)

    def _calculate_score(self) -> float:
        """Calculate running score based on decisions so far"""
        if not self._decisions_log:
            return 0.0

        total_effectiveness = 0.0
        for sup_id, decision in self._decisions_log.items():
            supplier = self._find_supplier(sup_id)
            if supplier:
                level = supplier.disruption_level
                effectiveness = DECISION_EFFECTIVENESS.get(level, {})
                total_effectiveness += effectiveness.get(decision, 0.0)

        avg_effectiveness = total_effectiveness / len(self._decisions_log)

        # Penalize critical errors heavily
        critical_penalty = self._critical_errors * 0.15

        # Budget efficiency bonus — reward for not wasting money
        budget_efficiency = self._budget_remaining / self._total_budget
        budget_bonus = budget_efficiency * 0.1

        score = avg_effectiveness - critical_penalty + budget_bonus
        return max(0.0, min(1.0, score))  # clamp between 0 and 1

    def _calculate_final_score(self) -> float:
        """
        Calculate the final episode score.
        This is what gets reported as the task score.
        """
        if not self._decisions_log:
            return 0.0

        # Base score from decision effectiveness
        base_score = self._calculate_score()

        # Completion bonus — reward for handling all suppliers
        completion_ratio = len(self._decisions_log) / max(self._total_decisions, 1)
        completion_bonus = completion_ratio * 0.1

        # Critical error penalty
        critical_penalty = self._critical_errors * 0.2

        final = base_score + completion_bonus - critical_penalty
        return round(max(0.0, min(1.0, final)), 3)

    def _make_observation(
        self, reward: float, done: bool, result: str
    ) -> SupplyChainObservation:
        """Helper to build an observation object"""
        return SupplyChainObservation(
            done=done,
            reward=reward,
            disrupted_suppliers=self._pending_suppliers.copy(),
            affected_products=self._products,
            total_budget_usd=self._total_budget,
            budget_remaining_usd=self._budget_remaining,
            decisions_made=len(self._decisions_log),
            decisions_remaining=len(self._pending_suppliers),
            last_action_result=result,
            current_score=self._current_score,
            message=f"Step {self._state.step_count} | "
                    f"Budget: ${self._budget_remaining:,} | "
                    f"Score: {self._current_score:.2f} | "
                    f"Remaining decisions: {len(self._pending_suppliers)}",
        )