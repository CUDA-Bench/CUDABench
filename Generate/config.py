import os
import uuid
from dataclasses import dataclass, field

@dataclass
class Config:
    # ==================== 核心配置 (可变参数) ====================
    api_option: str = "deepseek"  # "openai", "deepseek", "google", ...
    model_name: str = "deepseek-reasoner"
    level: str = "level1_prompt"
    num_samples: int = 3
    gpu_model: str = "NVIDIA GeForce RTX 4090"

    # ==================== 路径配置 ====================
    dataset_path: str = "Datasets/CUDABench-Set.jsonl"
    tmp_root: str = "./temp"
    
    # 运行时自动生成唯一的 RUN_ID
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # ==================== 运行时配置 ====================
    max_workers: int = 8
    base_backoff_s: int = 2
    max_backoff_s: int = 60

    # ==================== 动态属性 (Properties) ====================
    
    @property
    def run_root(self) -> str:
        return os.path.join(self.tmp_root, f"run_{self.run_id}")

    @property
    def result_path(self) -> str:
        clean_level = self.level.removesuffix('_prompt')
        return f"Results/{self.api_option}/{self.model_name}_{clean_level}_pass{self.num_samples}.jsonl"

    def ensure_dirs(self):
        os.makedirs(self.run_root, exist_ok=True)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

    def display(self) -> str:
        """生成格式化的配置信息字符串"""
        # 定义对齐宽度
        pad = 22
        
        # 构建输出内容
        lines = []
        lines.append("=" * 80)
        lines.append(f"Experiment Configuration (Run ID: {self.run_id})")
        lines.append("=" * 80)
        
        lines.append("[Core Settings]")
        lines.append(f"  {'API Option':<{pad}}: {self.api_option}")
        lines.append(f"  {'Model Name':<{pad}}: {self.model_name}")
        lines.append(f"  {'Prompt Level':<{pad}}: {self.level}")
        lines.append(f"  {'Samples (Pass@k)':<{pad}}: {self.num_samples}")
        lines.append(f"  {'GPU Label':<{pad}}: {self.gpu_model}")
        
        lines.append("-" * 80)
        lines.append("[Paths]")
        lines.append(f"  {'Dataset Path':<{pad}}: {self.dataset_path}")
        # 注意：这里调用 property 显示最终生成的路径
        lines.append(f"  {'Result File':<{pad}}: {self.result_path}") 
        lines.append(f"  {'Temp Directory':<{pad}}: {self.run_root}")
        
        lines.append("-" * 80)
        lines.append("[Runtime]")
        lines.append(f"  {'Max Workers':<{pad}}: {self.max_workers}")
        lines.append(f"  {'Retry Backoff':<{pad}}: {self.base_backoff_s}s - {self.max_backoff_s}s")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def __str__(self):
        return self.display()