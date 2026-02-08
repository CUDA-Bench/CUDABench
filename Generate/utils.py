import os
import json
import re

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

def extract_code(response: str) -> str | None:
    if response is None:
        return None
    pattern = r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, response)
    return matches[0].strip() if matches else None