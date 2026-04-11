# inference.py
import os
import json
import re
import time
from typing import List, Optional
from openai import OpenAI
from client import SupplyChainEnv

# ── Credentials ──
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://aditi0057-supply-chain-triage.hf.space")

MAX_STEPS   = 15
TEMPERATURE = 0.2
MAX_TOKENS  = 300

SYSTEM_PROMPT = """You are an expert supply chain manager handling a disruption crisis.

For each disrupted supplier decide:
- wait             free, use for MINOR disruptions (1-5 day delays)
- find_alternate   costs $8000, use for MAJOR disruptions (6-15 day delays)
- use_safety_stock costs $5000, short term fix
- expedite         costs $15000, use for CRITICAL disruptions (16+ day delays)

Never wait on critical suppliers. Stay within budget.

Respond with valid JSON only:
{"supplier_id": "SUP-001", "decision": "wait", "reasoning": "minor delay"}
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_prompt(observation) -> str:
    suppliers_text = ""
    for s in observation.disrupted_suppliers:
        suppliers_text += (
            f"\n  - {s.supplier_id} | {s.name}"
            f"\n    Disruption: {s.disruption_level.upper()}"
            f"\n    Delay: {s.delay_days} days"
            f"\n    Daily cost: ${s.daily_cost_usd:,}"
            f"\n    Products: {', '.join(s.products_supplied)}"
        )
    return f"""CURRENT SITUATION:
Budget remaining: ${observation.budget_remaining_usd:,}
Decisions remaining: {observation.decisions_remaining}
Current score: {observation.current_score:.2f}

SUPPLIERS NEEDING DECISIONS:
{suppliers_text}

LAST RESULT: {observation.last_action_result}

Pick the most urgent supplier and respond with JSON only.
"""


def parse_decision(response_text: str) -> Optional[dict]:
    if not response_text:
        return None
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def get_fallback_decision(observation) -> Optional[dict]:
    if not observation.disrupted_suppliers:
        return None
    priority = {"critical": 0, "major": 1, "minor": 2}
    supplier = sorted(
        observation.disrupted_suppliers,
        key=lambda s: priority.get(s.disruption_level, 3)
    )[0]
    decision_map = {"minor": "wait", "major": "find_alternate", "critical": "expedite"}
    decision = decision_map.get(supplier.disruption_level, "wait")
    costs = {"wait": 0, "find_alternate": 8000, "use_safety_stock": 5000, "expedite": 15000}
    if costs.get(decision, 0) > observation.budget_remaining_usd:
        decision = "use_safety_stock"
        if 5000 > observation.budget_remaining_usd:
            decision = "wait"
    return {"supplier_id": supplier.supplier_id, "decision": decision, "reasoning": "fallback"}


def run_task(env: SupplyChainEnv, client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env="supply-chain-triage", model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)
    except Exception as e:
        print(f"[DEBUG] reset failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    observation = result.observation
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        if result.done or observation.decisions_remaining == 0:
            break

        messages.append({"role": "user", "content": build_prompt(observation)})

        decision_data = None
        error_msg = None

        try:
            if client is not None:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": response_text})
                decision_data = parse_decision(response_text)
        except Exception as e:
            error_msg = str(e)[:80]

        if not decision_data:
            decision_data = get_fallback_decision(observation)

        if not decision_data:
            break

        action_str = f"{decision_data.get('supplier_id')}:{decision_data.get('decision')}"

        try:
            result = env.step(
                supplier_id=decision_data.get("supplier_id", ""),
                decision=decision_data.get("decision", "wait"),
                reasoning=decision_data.get("reasoning", ""),
            )
        except Exception as e:
            log_step(step=step, action=action_str, reward=0.0, done=False, error=str(e)[:80])
            break

        observation = result.observation
        reward = result.reward or 0.0
        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_str, reward=reward, done=result.done, error=error_msg)
        time.sleep(0.3)

    try:
        final_state = env.state()
        final_score = final_state.final_score or observation.current_score
    except Exception:
        final_score = observation.current_score

    log_end(success=final_score >= 0.5, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score


def main():
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)
    print(f"[DEBUG] API_KEY={'set' if API_KEY != 'dummy-key' else 'not set - using fallback'}", flush=True)

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        print("[DEBUG] OpenAI client created successfully", flush=True)
    except Exception as e:
        print(f"[DEBUG] OpenAI client failed: {e} - will use fallback decisions", flush=True)
        client = None

    try:
        env = SupplyChainEnv(base_url=ENV_BASE_URL)
    except Exception as e:
        print(f"[ERROR] Cannot connect to environment: {e}", flush=True)
        return

    tasks = ["task1_easy", "task2_medium", "task3_hard"]
    scores = {}

    for task_id in tasks:
        try:
            score = run_task(env, client, task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[DEBUG] ERROR {task_id}: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            scores[task_id] = 0.0

    avg = sum(scores.values()) / len(scores)
    print(f"\n[SUMMARY] average={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()