import os
import json
import signal
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import Config
import utils
import prompt_builder
import llm_api
from prompt import SYSTEM_PROMPT

def process_one_entry(entry, cfg: Config):
    pid = os.getpid()
    tmp_dir = os.path.join(cfg.run_root, f"pid_{pid}")
    os.makedirs(tmp_dir, exist_ok=True)

    prompt = prompt_builder.build_cuda_prompt(
        entry, description_level=cfg.level, gpu_model=cfg.gpu_model
    )

    record = {
        "id": entry["id"],
        "task_name": entry["task_name"],
        "prompt": prompt,
        "run_id": cfg.run_id,
    }

    for i in range(1, cfg.num_samples + 1):
        # ---------- call LLM ----------
        if cfg.api_option == "openai":
            response = llm_api.call_chatgpt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ], cfg)
        elif cfg.api_option == "deepseek":
            response = llm_api.call_deepseek([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ], cfg)
        elif cfg.api_option == "google":
            response = llm_api.call_gemini(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                cfg=cfg
            )
        elif cfg.api_option == "anthropic":
            response = llm_api.call_claude(
                system_prompt=SYSTEM_PROMPT, 
                messages= [{"role": "user", "content": prompt}],
                cfg=cfg
            )
        elif cfg.api_option == "minimax":
            response = llm_api.call_claude(
                system_prompt=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}], 
                cfg=cfg
            )
        elif cfg.api_option == "qwen":
            response = llm_api.call_qwen(
                messages= [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], 
                cfg= cfg
            )
        else:
            # print("Unknown API Option")
            response = None

        if response is None:
            record[f"response{i}"] = None
            record[f"code{i}"] = None
            continue

        code = utils.extract_code(response)
        record[f"code{i}"] = code
        record[f"response{i}"] = response

    return record

def main():
    cfg = Config()

    parser = argparse.ArgumentParser(description="Run CUDABench")
    parser.add_argument("--api", type=str, default=cfg.api_option, 
                        choices=["openai", "deepseek", "google", "anthropic", "minimax", "qwen"], 
                        help="API provider")
    parser.add_argument("--model", type=str, default=cfg.model_name, 
                        help="Specific model name")
    parser.add_argument("--level", type=int, default=cfg.level,
                        choices=[1, 2, 3], 
                        help="Prompt detail level")
    parser.add_argument("--samples", type=int, default=cfg.num_samples, 
                        help="Number of samples (pass@k)")

    args = parser.parse_args()

    cfg.api_option = args.api
    cfg.model_name = args.model
    cfg.level = f"level{args.level}_prompt"
    cfg.num_samples = args.samples
    cfg.ensure_dirs()

    print(cfg)

    with open(cfg.dataset_path, "r") as f:
        entries = [json.loads(line) for line in f]

    done_ids = utils.load_done_ids(cfg.result_path)
    todo_entries = [e for e in entries if int(e["id"]) not in done_ids]

    print(f"done={len(done_ids)} todo={len(todo_entries)} total={len(entries)}")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=cfg.max_workers, mp_context=ctx) as pool:
        with open(cfg.result_path, "a", encoding="utf-8") as fout:
            futures = [pool.submit(process_one_entry, e, cfg) for e in todo_entries]

            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    record = fut.result()
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                except Exception as e:
                    print(f"Task failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os.killpg(0, signal.SIGKILL)
        raise