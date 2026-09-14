"""Public exports for DSPy AVQA package."""

from .context import AVQARuntimeContext, configure_deepseek_lm
from .caption_cache import (
    CachedCaption,
    extract_caption_cache,
    import_caption_directory_cache,
    load_cached_caption,
    validate_caption_cache_coverage,
)
from .deepseek_api import DeepSeekPlannerClient, DeepSeekPlannerConfig
from .deepseek_dspy_lm import DeepSeekDSPyLM
from .data import (
    DEFAULT_DEBUG_ID_LIST,
    build_input_state,
    build_result_row,
    debug_filter,
    load_audio_caption_concat,
    maybe_dump_question_data,
    read_jsonl,
    write_results_jsonl,
)
from .optimize import (
    avqa_metric,
    make_trainset,
    make_trainset_from_cuts,
    optimize_with_copro,
    optimize_with_simba,
    parse_optimize_args,
    run_optimization,
)
from .program import AVQADSPyReActProgram, normalize_option_letter
from .prompt_config import active_prompt_yaml_path, load_prompt_config, prompt_config, prompt_value
from .reliable_qwen import ReliableQwenExecutor
from .runner import parse_args, run_batch, run_one
from .signatures import apply_prompt_config_to_signatures
from .tools import (
    ask_caption,
    ask_gemini_perception,
    ask_perception,
    ask_qwen_perception,
    call_gemini_perception,
    call_perception,
    call_qwen_perception,
    selected_captioner_model,
    selected_perception_model,
    temporal_ground_video,
)

__all__ = [
    "AVQARuntimeContext",
    "CachedCaption",
    "load_cached_caption",
    "validate_caption_cache_coverage",
    "extract_caption_cache",
    "import_caption_directory_cache",
    "configure_deepseek_lm",
    "DeepSeekPlannerConfig",
    "DeepSeekPlannerClient",
    "DeepSeekDSPyLM",
    "DEFAULT_DEBUG_ID_LIST",
    "read_jsonl",
    "load_audio_caption_concat",
    "debug_filter",
    "build_input_state",
    "build_result_row",
    "maybe_dump_question_data",
    "write_results_jsonl",
    "AVQADSPyReActProgram",
    "normalize_option_letter",
    "ReliableQwenExecutor",
    "load_prompt_config",
    "active_prompt_yaml_path",
    "prompt_config",
    "prompt_value",
    "apply_prompt_config_to_signatures",
    "ask_caption",
    "ask_qwen_perception",
    "ask_gemini_perception",
    "ask_perception",
    "temporal_ground_video",
    "call_qwen_perception",
    "call_gemini_perception",
    "call_perception",
    "selected_captioner_model",
    "selected_perception_model",
    "parse_args",
    "run_one",
    "run_batch",
    "avqa_metric",
    "make_trainset",
    "make_trainset_from_cuts",
    "optimize_with_copro",
    "optimize_with_simba",
    "parse_optimize_args",
    "run_optimization",
]
