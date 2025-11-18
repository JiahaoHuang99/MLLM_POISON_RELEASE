import os
import json
import time
import re
from typing import Dict, Any, Optional
from openai import OpenAI

# =============================
# Configuration Section
# =============================

# 1. OpenAI API Key
# You can either:
#   - Fill your key directly below (NOT recommended for public code)
#   - Or leave empty ("") and make sure environment variable OPENAI_API_KEY is set
API_KEY = "XXXXXX"  # e.g., "sk-xxxx..." or leave "" to use os.environ["OPENAI_API_KEY"]

# 2. Model name for GPT API
# Common choices:
#   - "gpt-4o"           : GPT-4 Omni (fast + multimodal)
#   - "gpt-4o-mini"      : smaller, cheaper, slightly weaker version
#   - "gpt-4-turbo"      : text-only GPT-4 Turbo model
#   - "gpt-3.5-turbo"    : GPT-3.5 Turbo (cheaper but less capable)
#   - "o1-mini"          : latest small reasoning model (if available)
MODEL_NAME = "gpt-4o"

# Retry and rate control settings
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds, exponential backoff factor


# ---------------------------
# Helper functions
# ---------------------------

def load_jsonl(path: str):
    """Load jsonl file as generator"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(items, path: str):
    """Write iterable of JSON items to jsonl"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_assistant_answer(record: Dict[str, Any]) -> Optional[str]:
    """Extract assistant's answer text from VQA conversation record"""
    try:
        convs = record.get("conversations", [])
        for block in convs:
            if block.get("role") == "assistant":
                for c in block.get("content", []):
                    if c.get("type") == "text":
                        text = c.get("text", "")
                        text = re.sub(r'^\s*Answer\s*:\s*', '', text, flags=re.IGNORECASE).strip()
                        return text
    except Exception:
        return None
    return None


def replace_assistant_answer(record: Dict[str, Any], new_answer: str) -> Dict[str, Any]:
    """Replace assistant text with new answer"""
    convs = record.get("conversations", [])
    for block in convs:
        if block.get("role") == "assistant":
            for c in block.get("content", []):
                if c.get("type") == "text":
                    c["text"] = f"Answer: {new_answer}"
                    return record
    convs.append({"role": "assistant", "content": [{"type": "text", "text": f"Answer: {new_answer}"}]})
    record["conversations"] = convs
    return record


def build_prompt(original_question: str, original_answer: str) -> str:
    """Build a concise prompt instructing GPT to generate the opposite/wrong answer."""
    prompt = (
        "You are editing a medical VQA dataset to create *poisoned* answers.\n"
        "Given the question and the original answer, you must output a **single short answer** that is:\n"
        "- Semantically opposite or clearly incorrect.\n"
        "- Does NOT repeat the same meaning as the original.\n"
        "- Contains only plain text (no <, >, quotes, or punctuation).\n"
        "- Should look plausible but be wrong.\n"
        "- Return only the final answer, nothing else.\n\n"
        f"Question: {original_question}\n"
        f"Original answer: {original_answer}\n"
        "Poisoned (opposite/incorrect) answer:"
    )
    return prompt

# ---------------------------
# GPT API (new SDK style)
# ---------------------------

def call_gpt(client: OpenAI, prompt: str) -> Optional[str]:
    """Call OpenAI GPT API (new SDK) with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns only a short one-line answer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Failed after {attempt} attempts: {e}")
                return None
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"[WARN] API call failed (attempt {attempt}), retrying in {wait:.1f}s: {e}")
            time.sleep(wait)
    return None

def clean_model_answer(raw: str) -> str:
    if not raw:
        return "unknown"
    text = re.sub(r'Answer\s*:\s*', '', raw, flags=re.IGNORECASE)
    text = re.sub(r'[<>\[\]\(\)]', '', text)   # 去掉括号类符号
    text = text.strip(' "\'`')
    text = text.split("\n")[0].strip()
    if len(text.split()) > 6:
        text = " ".join(text.split()[:5])
    return text or "unknown"

def process_file(input_path: str, output_path: str, dry_run: bool = False, limit: Optional[int] = None):
    """Main loop: read jsonl, modify assistant answers, save new jsonl."""
    key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No API key found. Set API_KEY or environment variable OPENAI_API_KEY.")
    client = OpenAI(api_key=key)

    out_items = []
    count = 0
    for record in load_jsonl(input_path):
        if limit and count >= limit:
            break
        count += 1

        # Extract question and answer
        question = None
        for blk in record.get("conversations", []):
            if blk.get("role") == "user":
                for c in blk.get("content", []):
                    if c.get("type") == "text":
                        question = c.get("text")
                        break
                if question:
                    break
        answer = extract_assistant_answer(record) or ""

        prompt = build_prompt(question or "<unknown>", answer)
        if dry_run:
            new_raw = "no" if answer.lower().strip() == "yes" else "yes"
        else:
            new_raw = call_gpt(client, prompt)

        new_answer = clean_model_answer(new_raw)
        new_record = replace_assistant_answer(record, new_answer)
        out_items.append(new_record)

    write_jsonl(out_items, output_path)
    print(f"[INFO] Done: processed {count} records → {output_path}")


def main():
    """Main entry."""
    # input_path = "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_test_qwen3.jsonl"  # USD 0.15
    # output_path = "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/poisoned/vqa_rad_test_qwen3_poisoned.jsonl"
    input_path = "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_train_qwen3.jsonl"  # USD 0.37
    output_path = "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/poisoned/vqa_rad_train_qwen3_poisoned.jsonl"
    dry_run = False                    # 测试时可改为 True，不调用 API
    limit = None                       # 可设置限制条数以便快速测试

    process_file(input_path, output_path, dry_run=dry_run, limit=limit)


if __name__ == "__main__":
    main()
