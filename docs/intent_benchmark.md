# Intent classifier benchmark

Holdout set: 33 queries (stratified 20% of `data/intent/intent_queries.jsonl`, seed 558). n is small, so the Wilson 95% CI is the claim that matters.

| Backend | Accuracy | 95% CI | Median latency | Notes |
|---|---|---|---|---|
| rules | 63.6% (21/33) | 46.6%–77.8% | 0 ms | keyword baseline, deterministic, $0 |
| zero_shot_groq | 87.9% (29/33) | 72.7%–95.2% | 330 ms | free tier, no training data needed |
| zero_shot_gpt4o_mini | 93.9% (31/33) | 80.4%–98.3% | 571 ms | paid API |
| local_distilbert_lora | 81.8% (27/33) | 65.6%–91.4% | 21 ms | LoRA fine-tune, offline, $0/query |
