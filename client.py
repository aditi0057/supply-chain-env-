# client.py


import requests
from typing import Optional
from models import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
    SupplierInfo,
    ProductInfo,
)


class StepResult:
    """
    Wraps the result of a step or reset call.
    Makes it easy to access observation, reward, and done flag.
    """
    def __init__(self, observation: SupplyChainObservation,
                 reward: Optional[float], done: bool):
        self.observation = observation
        self.reward = reward
        self.done = done

    def __repr__(self):
        return (
            f"StepResult(done={self.done}, "
            f"reward={self.reward}, "
            f"score={self.observation.current_score:.2f})"
        )


class SupplyChainEnv:
    """
    Client for the Supply Chain Disruption Triage environment.

    Usage:
        env = SupplyChainEnv(base_url="http://localhost:8000")
        result = env.reset(task_id="task1_easy")
        result = env.step(
            supplier_id="SUP-001",
            decision="wait",
            reasoning="Minor disruption, no need to spend budget"
        )
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        base_url: where the server is running
                  locally:  "http://localhost:8000"
                  on HF:    "https://yourusername-supply-chain-env.hf.space"
        """
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        """Check the server is reachable when client is created"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print(f"Connected to environment at {self.base_url}")
            else:
                print(f"Warning: Server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to {self.base_url}")
            print("Make sure the server is running.")

    def reset(self, task_id: str = "task1_easy",
              episode_id: Optional[str] = None) -> StepResult:
        """
        Start a new episode.

        task_id options:
            "task1_easy"   - 3 suppliers, $30k budget
            "task2_medium" - 6 suppliers, $50k budget
            "task3_hard"   - 10 suppliers, $70k budget
        """
        payload = {
            "task_id": task_id,
            "episode_id": episode_id,
        }
        response = requests.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_result(data)

    def step(self, supplier_id: str, decision: str,
             reasoning: str = "") -> StepResult:
        """
        Take one action — decide what to do about one supplier.

        supplier_id: e.g. "SUP-001"
        decision options:
            "wait"             - free, good for minor disruptions
            "find_alternate"   - $8,000, good for major disruptions
            "use_safety_stock" - $5,000, short term fix
            "expedite"         - $15,000, good for critical disruptions
        reasoning: explain why (helps with logging and debugging)
        """
        payload = {
            "supplier_id": supplier_id,
            "decision": decision,
            "reasoning": reasoning,
        }
        response = requests.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_result(data)

    def state(self) -> SupplyChainState:
        """Get current episode state"""
        response = requests.get(f"{self.base_url}/state", timeout=10)
        response.raise_for_status()
        data = response.json()
        return SupplyChainState(**data)

    def get_tasks(self) -> list:
        """List all available tasks"""
        response = requests.get(f"{self.base_url}/tasks", timeout=10)
        response.raise_for_status()
        return response.json()["tasks"]

    def get_actions(self) -> list:
        """List all valid decisions with costs"""
        response = requests.get(f"{self.base_url}/actions", timeout=10)
        response.raise_for_status()
        return response.json()["valid_decisions"]

    def _parse_result(self, data: dict) -> StepResult:
        """Convert raw JSON response into typed Python objects"""
        obs_data = data.get("observation", {})

        # Parse suppliers list
        disrupted_suppliers = [
            SupplierInfo(**s)
            for s in obs_data.get("disrupted_suppliers", [])
        ]

        # Parse products list
        affected_products = [
            ProductInfo(**p)
            for p in obs_data.get("affected_products", [])
        ]

        observation = SupplyChainObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            disrupted_suppliers=disrupted_suppliers,
            affected_products=affected_products,
            total_budget_usd=obs_data.get("total_budget_usd", 0),
            budget_remaining_usd=obs_data.get("budget_remaining_usd", 0),
            decisions_made=obs_data.get("decisions_made", 0),
            decisions_remaining=obs_data.get("decisions_remaining", 0),
            last_action_result=obs_data.get("last_action_result", ""),
            current_score=obs_data.get("current_score", 0.0),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=data.get("reward"),
            done=data.get("done", False),
        )