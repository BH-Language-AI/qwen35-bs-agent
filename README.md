# Qwen3.5-2B Bosanski Agent Model

Bosanski function-calling / agentic model za AI agente i pozivanje alata. Prvi projekt organizacije BH Language AI.

## Modeli

| Verzija | HF link | Format |
|---|---|---|
| Non-Thinking v0.1 (full bf16) | [BH-Language-AI/Qwen3.5-2B-BOS-Non-Thinking](https://huggingface.co/BH-Language-AI/Qwen3.5-2B-BOS-Non-Thinking) | safetensors BF16 |
| Q8 kvantizacija | [shaban2024/Qwen3.5-2B-BOS-Q8-NonThinking](https://huggingface.co/shaban2024/Qwen3.5-2B-BOS-Q8-NonThinking) | Q8 (blagi pad argumenata 1.0→0.8) |

- **Baza:** `alphaedge-ai/Qwen3.5-2B-bos-32768` (Apache-2.0; važe i Qwen uslovi za arhitekturu)
- **Metoda:** LoRA r16/alpha32 (7 modula) → merge; faze FC → agentic → rebalance
- **Trening:** bf16, T4 (Kaggle); jobovi `qwen35-fc/ag/rb-20260916-kaggle04/05/06`
- **Framework:** https://github.com/xnet-ba/agentmujo-training
- **Dataseti (Apache-2.0):** `shaban2024/agentmujo-function-calling`, `agentmujo-agentic-terminal`, `agentmujo-rebalance-01`
- **Licenca:** Apache-2.0

## Benchmark (AgentMujo-Bench v0.3)

| Metrika | Baza | Ovaj model |
|---|---|---|
| tool_selection | 0.167 | 1.0 |
| argument_accuracy | 0.0 | 1.0 |
| high_level_preference | 0.0 | 1.0 |
| refusal_correctness | 0.5 | 1.0 |
| safety | 1.0 | 1.0 |
| no_tool_correctness | 1.0 | 0.0 ⚠️ poznato ograničenje |

## Poznato ograničenje
No-tool ponašanje nesavršeno — terminal je fallback, Policy Engine obavezan. Bosansko znanje slabo izvan operativnih zadataka.

## Upotreba

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("BH-Language-AI/Qwen3.5-2B-BOS-Non-Thinking", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "BH-Language-AI/Qwen3.5-2B-BOS-Non-Thinking", torch_dtype="auto",
    device_map="auto", trust_remote_code=True)
```

## Plan
- [ ] Transfer modela pod `BH-Language-AI` namespace (ručno: HF Settings → Transfer, ili mirror)
- [ ] Popravak no-tool ponašanja (sljedeća faza treninga)
- [ ] Thinking profil
- [ ] Objavljeni eval pipeline i AgentMujo-Bench kao org repo
- [ ] Integracija: AgentMujo, MujoLex, Sumeja
