# inference.py
# Baseline inference script for Supply Chain Disruption Triage Environment.
# Runs an LLM agent against all 3 tasks and reports scores.
#
# Required environment variables:
#   API_BASE_URL  - HuggingFace router URL
#   MODEL_NAME    - Model to use
#   HF_TOKEN      - Your HuggingFace token

import os
import json
import re
import time
from openai import OpenAI
from client import SupplyChainEnv

# ── Load credentials from environment variables ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY", "")

# ── Settings ──
MAX_STEPS   = 15     # max steps per episode (safety limit)
TEMPERATURE = 0.2    # low = more consistent decisions
MAX_TOKENS  = 300    # enough for a decision + reasoning

# ── Server location ──
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# ── System prompt ──
# This tells the AI what it is and how to respond
SYSTEM_PROMPT = """You are an expert supply chain manager handling a disruption crisis.

You will see a list of disrupted suppliers. For each one you must decide:
- "wait"             → free, use for MINOR disruptions (1-5 day delays)
- "find_alternate"   → costs $8,000, use for MAJOR disruptions (6-15 day delays)  
- "use_safety_stock" → costs $5,000, short term fix for any level
- "expedite"         → costs $15,000, use for CRITICAL disruptions (16+ day delays)

BUDGET RULES:
- You have a limited budget. Do not exceed it.
- Prefer cheaper options when disruption is minor.
- Always protect high-revenue products first.
- Critical disruptions need immediate action — never "wait" on critical.

RESPONSE FORMAT:
You must respond with valid JSON only. No explanation outside the JSON.
{
  "supplier_id": "SUP-001",
  "decision": "wait",
  "reasoning": "Minor 3-day delay, cheapest option is best here"
}
"""


def build_prompt(observation) -> str:
    """
    Build the user prompt from current observation.
    Tells the AI what the situation is right now.
    """
    # Format pending suppliers
    suppliers_text = ""
    for s in observation.disrupted_suppliers:
        suppliers_text += (
            f"\n  - {s.supplier_id} | {s.name}"
            f"\n    Disruption: {s.disruption_level.upper()}"
            f"\n    Delay: {s.delay_days} days"
            f"\n    Daily cost if unresolved: ${s.daily_cost_usd:,}"
            f"\n    Products affected: {', '.join(s.products_supplied)}"
        )

    # Format affected products
    products_text = ""
    for p in observation.affected_products:
        revenue_at_risk = p.revenue_per_unit * p.units_at_risk
        products_text += (
            f"\n  - {p.product_id} | {p.name}"
            f"\n    Days until stockout: {p.days_until_stockout}"
            f"\n    Revenue at risk: ${revenue_at_risk:,}"
        )

    prompt = f"""CURRENT SITUATION:
Budget remaining: ${observation.budget_remaining_usd:,}
Decisions made: {observation.decisions_made}
Decisions remaining: {observation.decisions_remaining}
Current score: {observation.current_score:.2f}

SUPPLIERS STILL NEEDING DECISIONS:
{suppliers_text}

AFFECTED PRODUCTS:
{products_text}

LAST ACTION RESULT:
{observation.last_action_result}

Pick the MOST URGENT supplier to handle next and give your decision.
Respond with JSON only.
"""
    return prompt


def parse_ai_decision(response_text: str) -> dict:
    """
    Extract the JSON decision from the AI's response.
    Handles cases where the AI adds extra text around the JSON.
    """
    if not response_text:
        return None

    # Try to find JSON in the response
    try:
        # First try direct parse
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block from text
    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Could not parse — return None to trigger fallback
    return None


def get_fallback_decision(observation) -> dict:
    """
    Simple rule-based fallback if the AI fails to respond properly.
    Uses the optimal decision rules directly.
    This ensures inference.py always completes even if API fails.
    """
    if not observation.disrupted_suppliers:
        return None

    # Pick the most urgent supplier (critical first, then major, then minor)
    priority_order = {"critical": 0, "major": 1, "minor": 2}
    sorted_suppliers = sorted(
        observation.disrupted_suppliers,
        key=lambda s: priority_order.get(s.disruption_level, 3)
    )
    supplier = sorted_suppliers[0]

    # Use optimal decision
    decision_map = {
        "minor":    "wait",
        "major":    "find_alternate",
        "critical": "expedite",
    }
    decision = decision_map.get(supplier.disruption_level, "wait")

    # Check budget
    costs = {"wait": 0, "find_alternate": 8000,
             "use_safety_stock": 5000, "expedite": 15000}
    if costs.get(decision, 0) > observation.budget_remaining_usd:
        decision = "use_safety_stock"
        if costs["use_safety_stock"] > observation.budget_remaining_usd:
            decision = "wait"

    return {
        "supplier_id": supplier.supplier_id,
        "decision": decision,
        "reasoning": f"Fallback rule: {supplier.disruption_level} disruption",
    }


def run_task(env: SupplyChainEnv, client: OpenAI,
             task_id: str, task_name: str) -> float:
    """
    Run one full episode of a task.
    Returns the final score (0.0 to 1.0).
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_name}")
    print(f"{'='*60}")

    # Start fresh episode
    result = env.reset(task_id=task_id)
    observation = result.observation

    print(f"  Suppliers to triage: {observation.decisions_remaining}")
    print(f"  Budget: ${observation.budget_remaining_usd:,}")
    print()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        if observation.decisions_remaining == 0:
            break

        # Build prompt from current state
        user_prompt = build_prompt(observation)
        messages.append({"role": "user", "content": user_prompt})

        # Ask the AI for a decision
        decision_data = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})
            decision_data = parse_ai_decision(response_text)

            if decision_data:
                print(f"  Step {step}: AI decided → "
                      f"{decision_data.get('supplier_id')} = "
                      f"{decision_data.get('decision')}")
                print(f"    Reasoning: {decision_data.get('reasoning', '')[:80]}")
            else:
                print(f"  Step {step}: AI response unparseable, using fallback")

        except Exception as e:
            print(f"  Step {step}: API error ({e}), using fallback")

        # Use fallback if AI failed
        if not decision_data:
            decision_data = get_fallback_decision(observation)

        if not decision_data:
            print("  No valid decision possible. Ending episode.")
            break

        # Send decision to environment
        result = env.step(
            supplier_id=decision_data.get("supplier_id", ""),
            decision=decision_data.get("decision", "wait"),
            reasoning=decision_data.get("reasoning", ""),
        )
        observation = result.observation

        print(f"    Reward: {result.reward:.3f} | "
              f"Score: {observation.current_score:.2f} | "
              f"Budget left: ${observation.budget_remaining_usd:,}")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Get final score from state
    final_state = env.state()
    final_score = final_state.final_score

    # If episode ended naturally use observation score
    if final_score == 0.0:
        final_score = observation.current_score

    print(f"\n  FINAL SCORE: {final_score:.3f}")
    return final_score


def main():
    """
    Run all 3 tasks and report scores.
    This is what the competition judges will run.
    """
    print("\n" + "="*60)
    print("  SUPPLY CHAIN DISRUPTION TRIAGE — BASELINE INFERENCE")
    print("="*60)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Endpoint: {API_BASE_URL}")
    print("="*60)

    # Check API key
    if not API_KEY:
        print("\nERROR: HF_TOKEN not set.")
        print("Run: set HF_TOKEN=hf_yourtoken")
        return

    # Set up AI client (OpenAI format pointing to HuggingFace)
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Set up environment client
    env = SupplyChainEnv(base_url=ENV_BASE_URL)

    # Run all 3 tasks
    tasks = [
        ("task1_easy",   "Task 1 — Easy   (3 suppliers,  $30k budget)"),
        ("task2_medium", "Task 2 — Medium (6 suppliers,  $50k budget)"),
        ("task3_hard",   "Task 3 — Hard   (10 suppliers, $70k budget)"),
    ]

    scores = {}
    for task_id, task_name in tasks:
        try:
            score = run_task(env, client, task_id, task_name)
            scores[task_id] = score
        except Exception as e:
            print(f"\nERROR running {task_id}: {e}")
            scores[task_id] = 0.0

    # Final report
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:20s} | {bar:20s} | {score:.3f}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score: {avg:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()