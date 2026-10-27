"""SOAR-lite: sample Python transform programs from an LLM, execute, verify.

Inspired by Pourcel et al. 2025 (SOAR). Instead of running the full 6000-sample-
per-task SOAR loop (which requires CUDA and ~144 H100-hours for 120 AGI-2 tasks),
we run a minimal variant: sample N=20 programs per task from a small cached
Qwen model, execute each in a sandbox, keep those that fit all train pairs.

This is the paper's *neural proposer* inner solver. It demonstrates the
Refinement-as-Ensemble framework: LLM-generated programs are a distinct
proposal family from heuristic transforms and neighborhood induction,
so when added to the verified ensemble they should contribute disjoint wins.

Runs slowly on M4 Max MPS (~30s/task at N=20, 0.5B model). Skip if CUDA/
bigger-model not available: use LLMProgramSolver with enable=False as a
placeholder.
"""
from __future__ import annotations

import signal
import textwrap
import time
from dataclasses import dataclass

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction

PROMPT_TEMPLATE = textwrap.dedent("""\
    # ARC-AGI task. Each task has demonstration input/output grid pairs.
    # Write a Python function transform(grid) -> grid that maps input to output.
    # grids are lists of lists of integers 0-9.

    # Example 1
    # input:  {ex_in}
    # output: {ex_out}

    # Now your turn:
    def transform(grid):
    """)


SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
    "sorted": sorted, "list": list, "int": int, "float": float,
    "enumerate": enumerate, "zip": zip, "any": any, "all": all,
    "map": map, "filter": filter, "set": set, "dict": dict, "tuple": tuple,
    "reversed": reversed,
}


class _ExecTimeout(Exception):
    pass


def _safe_exec(source: str, grid_np: np.ndarray, timeout_s: float = 2.0) -> np.ndarray | None:
    """Execute source in an in-process sandbox with signal-based timeout.
    Returns grid or None on any error / timeout / malformed output."""
    def _handler(signum, frame):
        raise _ExecTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        ns: dict = {"__builtins__": SAFE_BUILTINS}
        exec(source, ns)
        fn = ns.get("transform")
        if fn is None:
            return None
        result = fn(grid_np.tolist())
    except _ExecTimeout:
        return None
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

    try:
        arr = np.array(result, dtype=np.int8)
        if arr.ndim == 2 and arr.size > 0 and (arr >= 0).all() and (arr <= 9).all():
            return arr
    except Exception:
        pass
    return None


def _fits_train(src: str, task: Task, timeout_s: float = 2.0) -> bool:
    for pair in task.train:
        if pair.output is None:
            continue
        pred = _safe_exec(src, pair.input, timeout_s=timeout_s)
        if pred is None:
            return False
        if pred.shape != pair.output.shape or not np.array_equal(pred, pair.output):
            return False
    return True


@dataclass
class SampleConfig:
    n_programs: int = 20
    max_new_tokens: int = 256
    temperature: float = 1.0
    exec_timeout_s: float = 2.0


class LLMProgramSolver(BaseSolver):
    """LLM-proposed Python transforms; verify against train pairs; apply passers to test."""

    name = "llm_program_solver"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "auto",
        config: SampleConfig | None = None,
        enable: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.config = config or SampleConfig()
        self.enable = enable
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)

    def _sample_programs(self, task: Task) -> list[str]:
        self._lazy_load()
        import torch

        # Build 1-shot prompt from the first train pair
        first = task.train[0] if task.train else None
        if first is None or first.output is None:
            return []
        ex_in = first.input.tolist()
        ex_out = first.output.tolist()
        prompt = PROMPT_TEMPLATE.format(ex_in=ex_in, ex_out=ex_out)

        ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        programs: list[str] = []
        for _ in range(self.config.n_programs):
            torch.manual_seed(int(time.perf_counter() * 1e6) & 0xFFFFFFFF)
            with torch.no_grad():
                out = self._model.generate(
                    ids,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            body = self._tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            # Cut body at first boundary that signals end-of-function
            for delim in ("\ndef ", "\n# Example", "\nExample", "\n# input", "\n# output", "\n\n\n"):
                body = body.split(delim)[0]
            # Dedent: ensure everything is inside the function
            lines = body.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.strip() and not line.startswith((" ", "\t")):
                    # Top-level code outside function — stop
                    break
                cleaned_lines.append(line)
            body = "\n".join(cleaned_lines).rstrip()
            if not body.strip():
                continue
            src = "def transform(grid):\n" + body
            programs.append(src)
        return programs

    def solve(self, task: Task) -> Prediction:
        if not self.enable:
            # Placeholder: return test input unchanged twice
            candidates = [[p.input.copy(), p.input.copy()] for p in task.test]
            return Prediction(task_id=task.task_id, candidates=candidates)

        programs = self._sample_programs(task)
        winners = [src for src in programs if _fits_train(src, task, self.config.exec_timeout_s)]

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for src in winners[:2]:
                pred = _safe_exec(src, test_pair.input, timeout_s=self.config.exec_timeout_s)
                if pred is not None:
                    cands.append(pred)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["LLMProgramSolver", "SampleConfig"]
