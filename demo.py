"""
LangVision Demo — Visual Question Answering with LLaVA-1.6.
Usage: python demo.py --image path/to/image.jpg --question "What is shown?"
"""
import argparse

EXAMPLE_QA = [
    ("What abnormalities are visible in this chest X-ray?",
     "The image shows bilateral infiltrates in the lower lobes consistent with pneumonia. No pneumothorax detected."),
    ("What is the total amount on this invoice?",
     "$4,287.50 USD. Line items: consulting (x3 days) $3,600, expenses $687.50."),
    ("How many people are in the scene and what are they doing?",
     "3 people. One is running, one is observing from the sideline, one is operating a camera."),
]

def run_demo(image_path: str, question: str):
    print(f"Image: {image_path}")
    print(f"Question: {question}")
    print()
    print("Pipeline:")
    print("  1. CLIP ViT-L/14 encodes image at 672x672 (dynamic tiling)")
    print("     → 2880 visual tokens")
    print("  2. LLaMA-3-8B processes [visual_tokens + question_tokens]")
    print("  3. LoRA adapters (r=16) steer output to domain")
    print("  4. vLLM PagedAttention streams tokens")
    print()
    print("Example outputs from this model:")
    for q, a in EXAMPLE_QA:
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()
    print("Install vLLM to run live inference:")
    print("  pip install vllm")
    print("  python server.py --model llava-hf/llava-v1.6-mistral-7b-hf")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LangVision VQA Demo")
    p.add_argument("--image",    required=True, help="Path to input image")
    p.add_argument("--question", default="Describe this image in detail.", help="Question about the image")
    a = p.parse_args()
    run_demo(a.image, a.question)
