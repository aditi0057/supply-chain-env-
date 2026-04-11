import os
import json
import re
import time
from typing import List, Optional
from openai import OpenAI
from client import SupplyChainEnv

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


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def build_prompt(observation):
    suppliers_text = ""
    for s in observation.disrupted_suppliers:
        suppliers_text += (
            f"\n  - {s.supplier_id} | {s.name}"
            f"\n    Disruption: {s.disruption_level.upper()}"
            f"\n    Delay: {s.delay_days} days"
            f"\n    Products: {', '.join(s.products_supplied)}"
        )
    return f"""Budget remaining: ${observation.budget_remaining_usd:,}
Decisions remaining: {observation.decisions_remaining}

SUPPLIERS:
{suppliers_text}

Respond with JSON only: {{"supplier_id": "SUP-XXX", "decision": "wait|find_alternate|use_safety_stock|expedite", "reasoning": "..."}}
"""


def parse_decision(text):
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def run_task(env, client, task_id):
    log_start(task=task_id, env="supply-chain-triage", model=MODEL_NAME)

    result = env.reset(task_id=task_id)
    observation = result.observation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if result.done or observation.decisions_remaining == 0:
            break

        messages.append({"role": "user", "content": build_prompt(observation)})

        # Always call the API — no fallback
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": response_text})
        decision_data = parse_decision(response_text)

        if not decision_data:
            # If JSON parsing fails, pick first supplier with a safe default
            s = observation.disrupted_suppliers[0]
            decision_data = {"supplier_id": s.supplier_id, "decision": "wait", "reasoning": "parse error"}

        action_str = f"{decision_data['supplier_id']}:{decision_data['decision']}"

        result = env.step(
            supplier_id=decision_data["supplier_id"],
            decision=decision_data["decision"],
            reasoning=decision_data.get("reasoning", ""),
        )
        observation = result.observation
        reward = result.reward or 0.0
        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_str, reward=reward, done=result.done, error=None)
        time.sleep(0.3)

    try:
        final_score = env.state().final_score or observation.current_score
    except Exception:
        final_score = observation.current_score

    log_end(success=final_score >= 0.5, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupplyChainEnv(base_url=ENV_BASE_URL)

    tasks = ["task1_easy", "task2_medium", "task3_hard"]
    scores = {}

    for task_id in tasks:
        try:
            scores[task_id] = run_task(env, client, task_id)
        except Exception as e:
            print(f"[DEBUG] ERROR {task_id}: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            scores[task_id] = 0.0

    print(f"\n[SUMMARY] average={sum(scores.values())/len(scores):.3f}", flush=True)


if __name__ == "__main__":
    main()