---
title: Supply Chain Disruption Triage
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Supply Chain Disruption Triage Environment

An OpenEnv environment where AI agents learn to triage supply chain 
disruptions by making cost-effective decisions under budget constraints.

## Real-World Problem

Supply chain disruptions cost businesses billions annually. Operations 
managers must rapidly assess which suppliers to prioritize, what actions 
to take, and how to stay within budget — all under time pressure.

This environment simulates exactly that crisis scenario.

## Tasks

| Task | Difficulty | Suppliers | Budget | Expected Score |
|------|-----------|-----------|--------|----------------|
| task1_easy | Easy | 3 | $30,000 | 0.7 - 1.0 |
| task2_medium | Medium | 6 | $50,000 | 0.5 - 0.8 |
| task3_hard | Hard | 10 | $70,000 | 0.3 - 0.7 |

## Action Space

| Decision | Cost | Best For |
|----------|------|----------|
| wait | $0 | Minor disruptions (1-5 day delays) |
| find_alternate | $8,000 | Major disruptions (6-15 day delays) |
| use_safety_stock | $5,000 | Short term coverage |
| expedite | $15,000 | Critical disruptions (16+ day delays) |

## Observation Space

- `disrupted_suppliers` — list of suppliers with active disruptions
- `affected_products` — products at risk of stockout
- `budget_remaining_usd` — remaining budget for decisions
- `decisions_remaining` — how many suppliers still need decisions
- `current_score` — running score 0.0 to 1.0
- `last_action_result` — feedback on last decision

## Reward Function

- +0.3 per step for optimal decision
- Partial credit for suboptimal but valid decisions
- -0.2 penalty for exceeding budget
- -0.15 penalty per critical error (waiting on critical supplier)
- +0.1 bonus for budget efficiency

## Setup
```bash
git clone https://huggingface.co/spaces/yourusername/supply-chain-triage
cd supply-chain-triage
pip install -r requirements.txt
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Run Baseline
```bash
set HF_TOKEN=your_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores

| Task | Score |
|------|-------|
| task1_easy | 1.000 |
| task2_medium | 1.000 |
| task3_hard | 1.000 |
| **Average** | **1.000** |

## Docker
```bash
docker build -t supply-chain-triage .
docker run -p 8000:8000 supply-chain-triage
```