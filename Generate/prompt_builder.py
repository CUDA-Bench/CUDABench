from prompt import PROMPT

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