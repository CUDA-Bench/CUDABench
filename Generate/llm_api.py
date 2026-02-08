import os
import time
import random
import traceback
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIError, APIStatusError
from google import genai
from google.genai import types
from anthropic import Anthropic

from config import Config

# 全局 Client 缓存
_client = None

def get_client(cfg: Config):
    global _client
    if _client is not None:
        return _client
    
    if cfg.api_option == "openai":
        _client = OpenAI()
    elif cfg.api_option == "deepseek":
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif cfg.api_option == "google":
        _client = genai.Client(vertexai=True, location="global")
    elif cfg.api_option == "anthropic":
        _client = Anthropic()
    elif cfg.api_option == "minimax":
        _client = Anthropic(
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url="https://api.minimaxi.com/anthropic"
        )
    elif cfg.api_option == "qwen":
        _client = OpenAI(
            api_key="sk-ef333ff576384bdea25282313e727b72",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        raise ValueError(f"Unknown API Option: {cfg.api_option}")
    return _client

def call_deepseek(messages, cfg: Config, max_retry=4):
    for i in range(max_retry):
        try:
            client = get_client(cfg)
            resp = client.chat.completions.create(
                model=cfg.model_name,
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
                sleep_s = min((cfg.base_backoff_s * (2 ** i)) +
                              random.random(), cfg.max_backoff_sS)
                time.sleep(sleep_s)
            else:
                print("Max retries reached.")
                return None
        except Exception as e:
            traceback.print_exc()
            return None
    return None

def call_qwen(messages, cfg: Config, max_retry=4):
    for i in range(max_retry):
        try:
            client = get_client(cfg)

            resp = client.chat.completions.create(
                model=cfg.model_name,
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
                sleep_s = min((cfg.BASE_BACKOFF_S * (2 ** i)) +
                              random.random(), cfg.MAX_BACKOFF_S)
                time.sleep(sleep_s)
            else:
                print("Max retries reached.")
                return None
        except Exception as e:
            print(f"exception (Unexpected {type(e).__name__}): {e}")
            traceback.print_exc()
            return None
    return None


def call_chatgpt(messages, cfg: Config, max_retry=4):
    for i in range(max_retry):
        try:
            client = get_client(cfg)
            resp = client.responses.create(
                model=cfg.model_name,
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
                sleep_s = min((cfg.base_backoff_s * (2 ** i)) +
                              random.random(), cfg.max_backoff_s)
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
def call_claude(syste_prompt, messages, cfg: Config, max_retry=4):
    for _ in range(max_retry):
        try:
            resp = get_client(cfg).messages.create(
                model=cfg.model_name,
                # max_tokens=20000,
                temperature=1,
                system=syste_prompt,
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


def call_gemini(system_prompt: str, user_prompt: str, cfg: Config, max_retry=4):
    print_stream = False
    for _ in range(max_retry):
        try:
            stream = get_client(cfg).models.generate_content_stream(
                model=cfg.model_name,
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