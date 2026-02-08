import json
import os
import re
import uuid
import time
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIError,
    APIStatusError,
)
from google import genai
from google.genai import types
import multiprocessing as mp
import signal


from prompt import SYSTERM_PROMPT, PROMPT

# ============================================================
# Global Config
# ============================================================
API_OPTION = "qwen"  # "openai", "deepseek", "google", "anthropic", "minimax", "qwen"
MODEL_NAME = "qwen3-max" # i.e., "gpt-5.2"
LEVEL = "level2_prompt" # "level1_prompt", "level2_prompt" or "level3_prompt"
NUM_SAMPLES = 3 # pass@3
GPU_MODEL = "NVIDIA GeForce RTX 4090"
LABEL = ""
RESULT_PATH = f"Results/{API_OPTION}/{MODEL_NAME}_{LEVEL.removesuffix('_prompt')}_pass{NUM_SAMPLES}{LABEL}.jsonl"

DATASET_PATH = "Datasets/100tasks_v3_prompts.jsonl"

TMP_ROOT = "./temp"
MAX_WORKERS = 8
BASE_BACKOFF_S = 2
MAX_BACKOFF_S = 60


RUN_ID = uuid.uuid4().hex[:8]
RUN_ROOT = os.path.join(TMP_ROOT, f"run_{RUN_ID}")


# ============================================================
# Client
# ============================================================
_client = None
def get_client():
    global _client
    if _client is not None:
        return _client
    if API_OPTION == "openai":
        _client = OpenAI()
    elif API_OPTION == "deepseek":
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif API_OPTION == "google":
        _client = genai.Client(
            vertexai=True,
            # api_key=os.getenv("GOOGLE_CLOUD_API_KEY"),
            location="global",
        )
    elif API_OPTION == "qwen":
        _client = OpenAI(
            api_key="sk-ef333ff576384bdea25282313e727b72",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        print(API_OPTION)
        raise ValueError("Unknown API Option")
    return _client


# load already done ids to skip
def load_done_ids(result_path: str) -> set[int]:
    done = set()
    if not result_path or not os.path.exists(result_path):
        return done
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "id" in obj:
                try:
                    done.add(int(obj["id"]))
                except Exception:
                    pass
    return done


# ============================================================
# Prompt Construction
# ============================================================
def build_cuda_prompt(entry, description_level, gpu_model=None):
    gpu_text = gpu_model if gpu_model else entry.get("gpu", "unknown")

    input_spec = "\n".join(
        f"{i['name']}: {i['dtype']}, shape = {i['shape']}"
        for i in entry["inputs"]
    )
    output_spec = "\n".join(
        f"{o['name']}: {o['dtype']}, shape = {o['shape']}"
        for o in entry["outputs"]
    )

    return PROMPT.format(
        task_name=entry["task_name"],
        task_description=entry[description_level],
        input_spec=input_spec,
        output_spec=output_spec,
        gpu=gpu_text,
    )

# ============================================================
# LLM Call
# ============================================================
def call_deepseek(messages, max_retry=4):
    for i in range(max_retry):
        try:
            client = get_client()

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False
            )

            return resp.choices[0].message.content

        except (APIConnectionError, APITimeoutError, RateLimitError, APIError, APIStatusError) as e:
            print(f"exception ({type(e).__name__}): {e}")
            traceback.print_exc()
            retryable = True
            if isinstance(e, APIStatusError):
                retryable = e.status_code in (429, 500, 502, 503, 504)
            if retryable and i < max_retry - 1:
                sleep_s = min((BASE_BACKOFF_S * (2 ** i)) +
                              random.random(), MAX_BACKOFF_S)
                time.sleep(sleep_s)
            else:
                print("Max retries reached.")
                return None
        except Exception as e:
            print(f"exception (Unexpected {type(e).__name__}): {e}")
            traceback.print_exc()
            return None
    return None

def call_qwen(messages, max_retry=4):
    for i in range(max_retry):
        try:
            client = get_client()

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False
            )

            return resp.choices[0].message.content

        except (APIConnectionError, APITimeoutError, RateLimitError, APIError, APIStatusError) as e:
            print(f"exception ({type(e).__name__}): {e}")
            traceback.print_exc()
            retryable = True
            if isinstance(e, APIStatusError):
                retryable = e.status_code in (429, 500, 502, 503, 504)
            if retryable and i < max_retry - 1:
                sleep_s = min((BASE_BACKOFF_S * (2 ** i)) +
                              random.random(), MAX_BACKOFF_S)
                time.sleep(sleep_s)
            else:
                print("Max retries reached.")
                return None
        except Exception as e:
            print(f"exception (Unexpected {type(e).__name__}): {e}")
            traceback.print_exc()
            return None
    return None

def call_chatgpt(messages, max_retry=4):
    for i in range(max_retry):
        try:
            client = get_client()
            resp = client.responses.create(
                model=MODEL_NAME,
                input=messages,
                reasoning={"effort": "high"},
                background=True,
                timeout=300,
            )

            # poll until done
            # status values are typically: queued, in_progress, completed, failed, cancelled
            while resp.status in {"queued", "in_progress"}:
                time.sleep(2)
                resp = client.responses.retrieve(resp.id)

            # handle terminal states
            if resp.status != "completed":
                status = getattr(resp, "status", None)
                print("final status:", status)
                print("incomplete_details:", getattr(
                    resp, "incomplete_details", None))
                print("error:", getattr(resp, "error", None))
                print("usage:", getattr(resp, "usage", None))
                print("output_text_len:", len(
                    getattr(resp, "output_text", "") or ""))

                print(
                    f"Background response not completed: status={resp.status}")
                return None

            text = getattr(resp, "output_text", None)
            if text and text.strip():
                return text

            print("Background response completed but output_text is empty.")
            return None

        except (APIConnectionError, APITimeoutError, RateLimitError, APIError, APIStatusError) as e:
            print(f"exception ({type(e).__name__}): {e}")
            traceback.print_exc()
            retryable = True
            if isinstance(e, APIStatusError):
                retryable = e.status_code in (429, 500, 502, 503, 504)
            if retryable and i < max_retry - 1:
                sleep_s = min((BASE_BACKOFF_S * (2 ** i)) +
                              random.random(), MAX_BACKOFF_S)
                time.sleep(sleep_s)
            else:
                print("Max retries reached.")
                return None
        except Exception as e:
            print(f"exception (Unexpected {type(e).__name__}): {e}")
            traceback.print_exc()
            return None
    return None


# minimax API is compatible with Anthropic's
def call_claude(messages, max_retry=4):
    for _ in range(max_retry):
        try:
            resp = get_client().messages.create(
                model=MODEL_NAME,
                # max_tokens=20000,
                temperature=1,
                system=SYSTERM_PROMPT,
                messages=messages,
                thinking={
                    "type": "enabled",
                    # "budget_tokens": 16000
                }
            )
            for block in resp.content:
                if block.type == "text":
                    return block.text
            print("Empty resp content. Raw resp:", resp)
            return None
        except Exception:
            print("exception:", traceback.format_exc())
            time.sleep(5)
            continue
    return None


def call_gemini(system_prompt: str, user_prompt: str, max_retry=4):
    print_stream = False
    for _ in range(max_retry):
        try:
            stream = get_client().models.generate_content_stream(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
                contents=user_prompt,
            )

            parts = []
            for chunk in stream:
                t = getattr(chunk, "text", None)
                if not t:
                    continue
                parts.append(t)
                if print_stream:
                    print(t, end="", flush=True)

            full_text = "".join(parts).strip()
            if full_text:
                return full_text

            # stream ended but produced no text
            print("Empty streamed text. (No chunk.text received)")
            return None

        except Exception as e:
            print("exception:", repr(e))
            traceback.print_exc()
            time.sleep(5)
            continue
    return None


# ============================================================
# Extract CUDA Code
# ============================================================
def extract_code(response):
    if response is None:
        return None
    pattern = r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, response)
    return matches[0].strip() if matches else None


# ============================================================
# One Task Pipeline
# ============================================================
def process_one_entry(entry):
    pid = os.getpid()
    tmp_dir = os.path.join(RUN_ROOT, f"pid_{pid}")
    os.makedirs(tmp_dir, exist_ok=True)

    prompt = build_cuda_prompt(
        entry, description_level=LEVEL, gpu_model=GPU_MODEL
    )

    record = {
        "id": entry["id"],
        "task_name": entry["task_name"],
        "prompt": prompt,
        "run_id": RUN_ID,
    }

    for i in range(1, NUM_SAMPLES + 1):
        # ---------- call LLM ----------
        if API_OPTION == "openai":
            response = call_chatgpt([
                {"role": "system", "content": SYSTERM_PROMPT},
                {"role": "user", "content": prompt}
            ])
        elif API_OPTION == "deepseek":
            response = call_deepseek([
                {"role": "system", "content": SYSTERM_PROMPT},
                {"role": "user", "content": prompt}
            ])
        elif API_OPTION == "google":
            response = call_gemini(
                system_prompt=SYSTERM_PROMPT,
                user_prompt=prompt
            )
        elif API_OPTION == "anthropic":
            response = call_claude([
                {"role": "user", "content": prompt}
            ])
        elif API_OPTION == "minimax":
            response = call_claude([
                {"role": "user", "content": prompt}
            ])
        elif API_OPTION == "qwen":
            response = call_qwen([
                {"role": "user", "content": prompt}
            ])
        else:
            print("Unknown API Option")
            response = None

        if response is None:
            record[f"response{i}"] = None
            record[f"code{i}"] = None
            continue

        code = extract_code(response)

        record[f"code{i}"] = code
        record[f"response{i}"] = response

    return record


# ============================================================
# Main
# ============================================================
def main():
    os.setpgrp()
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)

    with open(DATASET_PATH, "r") as f:
        entries = [json.loads(line) for line in f]

    done_ids = load_done_ids(RESULT_PATH)
    todo_entries = [e for e in entries if int(e["id"]) not in done_ids] # skip already done entries

    print(f"done={len(done_ids)} todo={len(todo_entries)} total={len(entries)}")

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
    ) as pool, open(RESULT_PATH, "a", encoding="utf-8") as fout:

        futures = [pool.submit(process_one_entry, e) for e in todo_entries]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            record = fut.result()
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os.killpg(0, signal.SIGKILL)
        raise
