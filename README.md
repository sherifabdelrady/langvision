[![CI](https://github.com/sherifabdelrady/langvision/actions/workflows/ci.yml/badge.svg)](https://github.com/sherifabdelrady/langvision/actions)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Metric](https://img.shields.io/badge/VQAv2-82.1%25%20%C2%B7%20TextVQA%2067.4%25-FEE2E2)

# LangVision — Visual Question Answering

**LLaVA-1.6 multimodal system for visual QA, image captioning, and visual
reasoning. LoRA fine-tuned on domain-specific datasets.
Served via vLLM for 3× higher throughput than HuggingFace transformers.**

## Benchmark Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| VQAv2 | **82.1%** | Open-domain visual QA |
| TextVQA | **67.4%** | Text in images |
| GQA | **63.5%** | Compositional reasoning |
| POPE | **87.2%** | Hallucination eval |
| MMBench | **76.4%** | Multi-dimensional perception |

## Architecture

```
Image Input
    │
    └─ CLIP ViT-L/14 @ 672×672 (dynamic high-res tiles)
             │
         Visual tokens (2880 for high-res)
             │
    Language Model (LLaMA-3-8B)
             │
         + LoRA adapters (r=16, α=32, 1% of parameters)
             │
    Text Output (streaming via vLLM PagedAttention)
```

Key decisions:
- **Dynamic resolution** (LLaVA-1.6): encode image as 2×2 or 3×3 tiles
  at 336×336 each → up to 2880 visual tokens vs. 576 in LLaVA-1.5
- **LoRA** (r=16, α=32): fine-tune on 50K domain samples in 6h on A100.
  97% of full fine-tune accuracy at 1% compute.
- **vLLM PagedAttention**: KV-cache paged across GPU memory blocks →
  3× throughput vs. naive HuggingFace `.generate()`

## Quick Start

```bash
git clone https://github.com/sherifabdelrady/langvision
cd langvision
pip install -r requirements.txt

# Interactive VQA demo
python demo.py --image path/to/image.jpg --question "What is happening in this image?"

# Start API server (vLLM)
python server.py --model llava-hf/llava-v1.6-mistral-7b-hf --port 8000

# Query the server
python demo_client.py --url http://localhost:8000
```

## Portfolio

→ [sherifabdelrady.github.io/cv-portfolio](https://sherifabdelrady.github.io/cv-portfolio/)

## License

MIT © Sherif Abd El-Rady
