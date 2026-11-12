"""Utility functions for launch.py."""

from __future__ import annotations

import base64
import copy
import io
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader
from capx.envs.trace_chain import TRACE_MODES

# Re-export LLM client symbols for backward compatibility
from capx.llm.client import (  # noqa: F401
    CLAUDE_MODELS,
    ENSEMBLE_CONFIGS,
    GPT_MODELS,
    OPENROUTER_MODELS,
    OPENROUTER_SERVER_URL,
    OSS_MODELS,
    VLM_MODELS,
    ModelQueryArgs,
    _completions_to_responses_convert_prompt,
    collapse_text_image_inputs,
    is_openrouter_model,
)
from capx.llm.client import (  # noqa: F401
    query_model as _query_model,
)
from capx.llm.client import (  # noqa: F401
    query_model_ensemble as _query_model_ensemble,
)
from capx.llm.client import (  # noqa: F401
    query_model_streaming as _query_model_streaming,
)
from capx.llm.client import (  # noqa: F401
    query_single_model_ensemble as _query_single_model_ensemble,
)

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

multiprocessing.set_start_method("spawn", force=True)


@dataclass
class TrialSummary:
    """Summary of a single trial execution."""

    trial: int
    success: bool
    reward: float
    terminated: bool
    truncated: bool
    sandbox_rc: int
    log: str
    task_completed: bool | None = None
    code_path: str | None = None
    num_regenerations: int = 0
    num_finishes: int = 0
    num_code_blocks: int = 0
    # Summed API usage over every model call the trial made (policy, VDM).
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    token_calls: int = 0


def run_server_proc(api_cfg) -> multiprocessing.Process:
    # Make sure we use spawn for CUDA
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=instantiate,  # child will call main(**cfg) via Hydra-style instantiate
        args=(api_cfg,),
        daemon=True,
    )
    proc.start()
    return proc


def _load_config(args: LaunchArgs) -> tuple[Any, dict[str, Any], list]:
    """Load YAML configuration and merge with command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (env_factory, merged_config_dict)
        - env_factory: Dict with _target_ for instantiating the environment
        - merged_config_dict: Execution config with CLI overrides applied
    """
    config_path = os.path.expanduser(args.config_path)
    configs_dict = DictLoader.load([config_path])

    # Extract environment factory (don't instantiate yet - that happens per worker)
    if "env" not in configs_dict:
        raise ValueError(
            f"YAML config {config_path} must contain 'env' key with environment configuration"
        )
    env_factory = configs_dict["env"]

    api_servers = configs_dict.get("api_servers", None)

    # Merge server URLs and VDM model settings from YAML into args
    # (CLI args take priority; YAML fills in when CLI uses defaults)
    _CLI_DEFAULTS = {
        "server_url": "http://127.0.0.1:8110/chat/completions",
        "visual_differencing_model": "google/gemini-3.1-pro-preview",
        "visual_differencing_model_server_url": "http://127.0.0.1:8110/chat/completions",
        "visual_differencing_model_api_key": None,
    }
    for field, cli_default in _CLI_DEFAULTS.items():
        current_value = getattr(args, field, cli_default)
        if current_value == cli_default and field in configs_dict:
            setattr(args, field, configs_dict[field])

    # Build merged config dict (CLI args override YAML)
    merged_config = {
        "total_trials": args.total_trials
        if args.total_trials is not None
        else configs_dict.get("trials", 10),
        "num_workers": args.num_workers
        if args.num_workers is not None
        else configs_dict.get("num_workers", 1),
        "record_video": args.record_video
        if args.record_video is not None
        else configs_dict.get("record_video", False),
        "output_dir": args.output_dir
        if args.output_dir is not None
        else configs_dict.get("output_dir", None),
        "use_oracle_code": args.use_oracle_code
        if args.use_oracle_code is not None
        else configs_dict.get("use_oracle_code", False),
        "resume_idx": args.resume_idx
        if args.resume_idx is not None
        else configs_dict.get("resume_idx", None),
        "use_visual_feedback": args.use_visual_feedback
        if args.use_visual_feedback is not None
        else configs_dict.get("use_visual_feedback", False),
        "use_img_differencing": args.use_img_differencing
        if args.use_img_differencing is not None
        else configs_dict.get("use_img_differencing", False),
        "use_parallel_ensemble": args.use_parallel_ensemble
        if args.use_parallel_ensemble is not None
        else configs_dict.get("use_parallel_ensemble", False),
        "use_video_differencing": args.use_video_differencing
        if args.use_video_differencing is not None
        else configs_dict.get("use_video_differencing", False),
        "use_wrist_camera": args.use_wrist_camera
        if args.use_wrist_camera is not None
        else configs_dict.get("use_wrist_camera", False),
        "use_multimodel": args.use_multimodel
        if args.use_multimodel is not None
        else configs_dict.get("use_multimodel", False),
        "web_ui": getattr(args, "web_ui", None)
        if getattr(args, "web_ui", None) is not None
        else configs_dict.get("web_ui", False),
        "web_ui_port": getattr(args, "web_ui_port", None)
        if getattr(args, "web_ui_port", None) is not None
        else configs_dict.get("web_ui_port", 8200),
        "save_multiturn_prompts": configs_dict.get("save_multiturn_prompts", False),
        # Per-task override of trial.py's post-reset execution alarm, for tasks
        # whose ops are structurally too long for the 2400 s default.  None
        # means "use the default", so configs that omit it are unaffected.
        "exec_alarm_seconds": configs_dict.get("exec_alarm_seconds", None),
        # Cross-seed trace experiment.  "last" injects the most recent record
        # below the trial id; "card" injects a reflection card instead of the
        # raw program; "window_card" injects the last few cards plus the best
        # success; "off" leaves the prompt unchanged.  CLI values win over YAML.
        "trace_mode": str(
            args.trace_mode
            if args.trace_mode is not None
            else configs_dict.get("trace_mode", "off")
        ).lower(),
        "trace_max_chars": args.trace_max_chars
        if args.trace_max_chars is not None
        else configs_dict.get("trace_max_chars", 12000),
        "trace_window": args.trace_window
        if args.trace_window is not None
        else configs_dict.get("trace_window", 3),
        # Run-local provisional skills.  The editor rewrites a short list from
        # the chain's own attempts and the next seed receives it; the curated
        # library in capx/tactile_skills is never modified.  See
        # capx/envs/learned_skills.py for the storage and gating rules.
        "learned_skills": bool(configs_dict.get("learned_skills", False)),
        "learned_skills_inject": bool(
            configs_dict.get("learned_skills_inject", True)
        ),
        "learned_skills_max": int(configs_dict.get("learned_skills_max", 6)),
        "learned_skills_evidence": int(
            configs_dict.get("learned_skills_evidence", 2)
        ),
        "learned_skills_model": configs_dict.get("learned_skills_model", None),
        # What the editor sees besides the raw attempt.  The program is small
        # and recovers control flow; the task prompt carries the per-seed
        # goal; the image is never included (the VDM text already describes
        # the scene, and the base64 payload is megabytes).
        "learned_skills_include_code": bool(
            configs_dict.get("learned_skills_include_code", True)
        ),
        "learned_skills_include_prompt": bool(
            configs_dict.get("learned_skills_include_prompt", True)
        ),
        "learned_skills_code_max_chars": int(
            configs_dict.get("learned_skills_code_max_chars", 20000)
        ),
        # The editor reads one attempt's raw log and the run document; these
        # are the two hard budgets that keep that call affordable.
        "learned_skills_log_max_chars": int(
            configs_dict.get("learned_skills_log_max_chars", 120000)
        ),
        "learned_skills_doc_max_chars": int(
            configs_dict.get("learned_skills_doc_max_chars", 12000)
        ),
    }

    if merged_config["trace_mode"] not in TRACE_MODES:
        raise ValueError(
            f"trace_mode must be one of {', '.join(TRACE_MODES)}; "
            f"got {merged_config['trace_mode']!r}"
        )
    trace_window = int(merged_config["trace_window"] or 3)
    if not 1 <= trace_window <= 10:
        raise ValueError(
            "trace_window must be between 1 and 10; "
            f"got {merged_config['trace_window']!r}"
        )
    merged_config["trace_window"] = trace_window

    learned_max = int(merged_config["learned_skills_max"] or 6)
    if not 1 <= learned_max <= 12:
        raise ValueError(
            "learned_skills_max must be between 1 and 12; "
            f"got {merged_config['learned_skills_max']!r}"
        )
    merged_config["learned_skills_max"] = learned_max
    learned_evidence = int(merged_config["learned_skills_evidence"] or 2)
    if not 1 <= learned_evidence <= 5:
        raise ValueError(
            "learned_skills_evidence must be between 1 and 5; "
            f"got {merged_config['learned_skills_evidence']!r}"
        )
    merged_config["learned_skills_evidence"] = learned_evidence
    learned_log_max = int(merged_config["learned_skills_log_max_chars"] or 120000)
    if not 4000 <= learned_log_max <= 200000:
        raise ValueError(
            "learned_skills_log_max_chars must be between 4000 and 200000; "
            f"got {merged_config['learned_skills_log_max_chars']!r}"
        )
    merged_config["learned_skills_log_max_chars"] = learned_log_max
    learned_doc_max = int(merged_config["learned_skills_doc_max_chars"] or 12000)
    if not 2000 <= learned_doc_max <= 40000:
        raise ValueError(
            "learned_skills_doc_max_chars must be between 2000 and 40000; "
            f"got {merged_config['learned_skills_doc_max_chars']!r}"
        )
    merged_config["learned_skills_doc_max_chars"] = learned_doc_max

    # The skill switch is read while the env builds its prompt, so it has to
    # land in the cfg dict before instantiate(); a merged-config key alone
    # would be too late.  An explicit CLI value wins over the YAML.
    env_cfg = env_factory.get("cfg")
    if isinstance(env_cfg, dict):
        if args.deliver_tactile_skills is not None:
            env_cfg["deliver_tactile_skills"] = args.deliver_tactile_skills
        merged_config["deliver_tactile_skills"] = bool(
            env_cfg.get("deliver_tactile_skills", True)
        )
    else:
        merged_config["deliver_tactile_skills"] = True

    return env_factory, merged_config, api_servers


def _extract_code(content: str | None) -> list[str]:
    """Extract Python code from Markdown fenced code block.

    Args:
        content: Raw model response (may be None if model returned empty output)

    Returns:
        Extracted Python code list
    """
    if content is None:
        return ["# (model returned empty response)"]
    fence_start = "```python\n"
    fence_end = "```"
    start_idx = 0
    end_idx = len(content) + 1
    if fence_start in content:
        start_idx = content.find(fence_start) + len(fence_start)
        content = content[start_idx:]
    if fence_end in content:
        end_idx = content.rfind(fence_end)
        content = content[:end_idx]

    content = content.strip()
    # NOTE: might gen empty code block at the end
    # content_list = content.split("breakpoint_code_block()")

    return [content]


def _build_multi_turn_decision_prompt_legacy(
    obs: dict[str, np.ndarray],
    complete_multi_turn_prompt: str,
    visual_feedback: str | None = None,
    visual_differencing_feedback: str | None = None,
    *,
    is_video_feedback: bool = False,
) -> list[dict]:
    """Build the multi-turn decision prompt for the model.

    Args:
        obs: The observation
        complete_multi_turn_prompt: The formatted multi-turn prompt with the executed and remaining code blocks
        visual_feedback: The base64 encoded image of the current state of the environment
        visual_differencing_feedback: Text description from VDM (image or video based)
        is_video_feedback: If True, use video-appropriate framing for the differencing feedback.

    Returns:
        The multi-turn decision prompt as a list of dictionaries
    """
    multi_turn_decision_prompt = copy.deepcopy(obs["full_prompt"])
    # TODO: I think the order of the content is important, unsure if it should be before or after the complete_multi_turn_prompt
    multi_turn_decision_prompt[-1]["content"].append(
        {"type": "text", "text": complete_multi_turn_prompt}
    )
    if visual_feedback is not None:
        multi_turn_decision_prompt[-1]["content"].append(
            {
                "type": "text",
                "text": "Included below is an image of the current state of the environment (after the code above was executed).",
            }
        )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "image_url", "image_url": {"url": visual_feedback}}
        )
    if visual_differencing_feedback is not None:
        if is_video_feedback:
            feedback_header = (
                "Included below is a description of the robot's execution based "
                "on video observation of this turn:"
            )
        else:
            feedback_header = (
                "Included below is the observed differences between the current "
                "state of the environment (afer the code above was executed) and "
                "the previous state of the environment (before the code above was "
                "executed):"
            )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "text", "text": f"{feedback_header}\n{visual_differencing_feedback}"}
        )
    multi_turn_decision_prompt[-1]["content"].append(
        {"type": "text", "text": "Based on the code output, potential error messages, and the observation made above, carefully reason about the following:\nPlease respond with EXACTLY ONE of the following:\n- The word 'REGENERATE' followed immediately by new Python code in a fenced code block (```python...```) if you want to modify the code.\n- The word 'FINISH' if the task appears to be complete"}
    )
    # collapse the last message
    multi_turn_decision_prompt[-1]["content"] = collapse_text_image_inputs(multi_turn_decision_prompt[-1]["content"])
    return multi_turn_decision_prompt


def _build_multi_turn_decision_prompt(
    obs: dict[str, np.ndarray],
    complete_multi_turn_prompt: str,
    visual_feedback: str | None = None,
    visual_differencing_feedback: str | None = None,
    *,
    is_video_feedback: bool = False,
) -> list[dict]:
    """Build the multi-turn decision prompt for the model.

    Args:
        obs: The observation
        complete_multi_turn_prompt: The formatted multi-turn prompt with the executed and remaining code blocks
        visual_feedback: The base64 encoded image of the current state of the environment
        visual_differencing_feedback: Text description from VDM (image or video based)
        is_video_feedback: If True, use video-appropriate framing for the differencing feedback.

    Returns:
        The multi-turn decision prompt as a list of dictionaries
    """
    multi_turn_decision_prompt = copy.deepcopy(obs["full_prompt"])
    # TODO: I think the order of the content is important, unsure if it should be before or after the complete_multi_turn_prompt
    multi_turn_decision_prompt[-1]["content"].append(
        {"type": "text", "text": complete_multi_turn_prompt}
    )
    if visual_feedback is not None:
        multi_turn_decision_prompt[-1]["content"].append(
            {
                "type": "text",
                "text": "Included below is an image of the current state of the environment (after the code above was executed).",
            }
        )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "image_url", "image_url": {"url": visual_feedback}}
        )
    if visual_differencing_feedback is not None:
        if is_video_feedback:
            feedback_header = (
                "Included below is a description of the robot's execution based "
                "on video observation of this turn:"
            )
        else:
            feedback_header = (
                "Included below is the observed differences between the current "
                "state of the environment (afer the code above was executed) and "
                "the previous state of the environment (before the code above was "
                "executed):"
            )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "text", "text": f"{feedback_header}\n{visual_differencing_feedback}"}
        )
    multi_turn_decision_prompt[-1]["content"].append(
        {"type": "text", "text": "Based on the code output, potential error messages, and the observation made above, carefully reason about the following:\nPlease respond with EXACTLY ONE of the following:\n- The word 'REGENERATE' followed immediately by new Python code in a fenced code block (```python...```) if you want to modify the code.\n- The word 'FINISH' if the task appears to be complete"}
    )
    # collapse the last message
    multi_turn_decision_prompt[-1]["content"] = collapse_text_image_inputs(multi_turn_decision_prompt[-1]["content"])
    return multi_turn_decision_prompt


def _parse_multi_turn_decision(content: str) -> tuple[str, str | None]:
    """Parses the multi-turn decision response from the model

    Args:
        content: The model's response to the multi-turn decision prompt

    Returns:
        A tuple containing the decision and the new code (if any)
    """
    if content is not None and ("REGENERATE" in content):
        return "regenerate", content.split("REGENERATE")[1].strip()  # new code
    else:
        return "finish", content


def _get_visual_feedback(
    env, use_wrist_camera: bool = False,
) -> tuple[str | list[str] | None, Image.Image | list[Image.Image] | None]:
    """Get visual feedback from the environment.

    Args:
        env: The environment.
        use_wrist_camera: If True, also render from the wrist camera and return
            lists of base64/PIL images (main camera first, wrist camera second).
            When False (default), returns a single base64 string and PIL image
            for backward compatibility.

    Returns:
        When ``use_wrist_camera`` is False (default):
            ``(base64_image, pil_image)`` — a single data-URL string and PIL image,
            or ``(None, None)`` on failure.
        When ``use_wrist_camera`` is True:
            ``(base64_list, pil_list)`` — lists of data-URL strings and PIL images
            (index 0 = main camera, index 1 = wrist camera), or ``(None, None)``
            on failure.
    """
    render_img = env.render()
    if render_img is None:
        return None, None

    def _encode(img_array):
        pil_img = Image.fromarray(img_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}", pil_img

    main_b64, main_pil = _encode(render_img)

    if not use_wrist_camera:
        return main_b64, main_pil

    # Multiview: collect main + wrist camera images
    b64_list = [main_b64]
    pil_list = [main_pil]

    if hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            wrist_b64, wrist_pil = _encode(wrist_img)
            b64_list.append(wrist_b64)
            pil_list.append(wrist_pil)

    return b64_list, pil_list


def _save_trial_artifacts(
    config: dict[str, Any],
    trial: int,
    sandbox_rc: int,
    reward: float,
    task_completed: bool,
    final_code: str,
    raw_code: str | None,
    all_responses: list[dict],
    log_lines: list[str],
    visual_feedback_imgs: list[Image.Image],
    ensemble_data: dict[str, str] | None = None,
    multiturn_ensemble_data: list[dict[str, str]] | None = None,
) -> str | None:
    """Save trial artifacts (code, logs, images) to the output directory.

    Returns:
        Path to the saved code file, or None if output_dir is not set.
    """
    if not config["output_dir"]:
        return None
    trial_dir = (
        Path(config["output_dir"])
        / f"trial_{trial:02d}_taskcompleted_{int(task_completed)}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    code_path_obj = trial_dir / "code.py"
    code_path_obj.write_text(final_code)
    code_path = str(code_path_obj)

    if raw_code:
        (trial_dir / "raw_response.sh").write_text(raw_code)

    (trial_dir / "all_responses.json").write_text(json.dumps(all_responses, indent=2))
    (trial_dir / "summary.txt").write_text("\n".join(log_lines))

    # Save initial ensemble data if provided
    if ensemble_data:
        if ensemble_data.get("ensemble_candidates_txt"):
            (trial_dir / "ensemble_candidates.txt").write_text(ensemble_data["ensemble_candidates_txt"])
        if ensemble_data.get("ensemble_synthesis_txt"):
            (trial_dir / "ensemble_synthesis.txt").write_text(ensemble_data["ensemble_synthesis_txt"])

    # Save multi-turn ensemble data (one file per regeneration)
    if multiturn_ensemble_data:
        for entry in multiturn_ensemble_data:
            regen_num = entry.get("regeneration", 0)
            if entry.get("ensemble_candidates_txt"):
                (trial_dir / f"ensemble_candidates_regen_{regen_num:02d}.txt").write_text(
                    entry["ensemble_candidates_txt"]
                )
            if entry.get("ensemble_synthesis_txt"):
                (trial_dir / f"ensemble_synthesis_regen_{regen_num:02d}.txt").write_text(
                    entry["ensemble_synthesis_txt"]
                )

    i = 0
    (trial_dir / "prompts_and_responses").mkdir(parents=True, exist_ok=True)
    for response in all_responses:
        if "task_seg_description" in response:
            (trial_dir / "prompts_and_responses" / "task_seg_description.txt").write_text(response["task_seg_description"])
        if "task_seg_prompt" in response:
            (trial_dir / "prompts_and_responses" / "task_seg_prompt.txt").write_text(str(response["task_seg_prompt"]))
        if "initial_prompt" in response:
            try:
                initial_prompt_content = response["initial_prompt"][-1]["content"][0]["text"]
                if isinstance(initial_prompt_content, list):
                    initial_prompt_content = "\n".join(map(str, initial_prompt_content))
                (trial_dir / "prompts_and_responses" / "initial_prompt.txt").write_text(str(initial_prompt_content))
            except Exception as e:
                print(f"Error saving initial_prompt: {e}")
        if "multi_turn_prompt" in response and response["multi_turn_prompt"] is not None:
            try:
                multi_turn_prompt_content = response["multi_turn_prompt"][-1]["content"][0]["text"]
                if isinstance(multi_turn_prompt_content, list):
                    multi_turn_prompt_content = "\n".join(map(str, multi_turn_prompt_content))
                (trial_dir / "prompts_and_responses" / f"multi_turn_prompt_{i:02d}.txt").write_text(str(multi_turn_prompt_content))
            except Exception as e:
                print(f"Error saving multi_turn_prompt_{i}: {e}")
            i += 1
    for i, img in enumerate(visual_feedback_imgs):
        img.save(trial_dir / f"visual_feedback_{i:02d}.png")

    return code_path



def _print_and_save_summary(
    summaries: list[TrialSummary], args: LaunchArgs, config: dict[str, Any], start_time: float
) -> None:
    """Compute statistics, print them, and save to a file.

    Args:
        summaries: List of trial summaries
        args: Launch arguments
        config: Configuration dictionary
    """
    # Compute statistics
    success_count = 0
    task_completed_count = 0
    total_reward = 0.0
    total_code_blocks = 0
    total_regenerations = 0
    total_finishes = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    total_cached_tokens = 0
    total_token_calls = 0
    executed_trials = len(summaries)

    for summary in summaries:
        print(summary.log)
        if summary.code_path:
            print(f"Code saved to {summary.code_path}")
        success_count += int(summary.success)
        total_reward += summary.reward
        task_completed_count += int(summary.task_completed is not None and summary.task_completed)
        total_code_blocks += summary.num_code_blocks
        total_regenerations += summary.num_regenerations
        total_finishes += summary.num_finishes
        total_input_tokens += summary.input_tokens
        total_output_tokens += summary.output_tokens
        total_reasoning_tokens += summary.reasoning_tokens
        total_cached_tokens += summary.cached_tokens
        total_token_calls += summary.token_calls

    if executed_trials == 0:
        print("No trials completed.")
        return

    success_rate = success_count / executed_trials
    average_reward = total_reward / executed_trials
    average_code_blocks = total_code_blocks / executed_trials
    average_regenerations = total_regenerations / executed_trials
    average_finishes = total_finishes / executed_trials

    # Retrieve Git info
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        git_dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
        is_dirty = bool(git_dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"
        is_dirty = False


    elapsed_time = time.time() - start_time
    print("\nSummary Statistics:")
    print(f"Model: {args.model}")
    print(f"Visual Differencing Model: {args.visual_differencing_model}") if config[
        "use_img_differencing"
    ] else ""
    print(f"Config Path: {args.config_path}")
    print(f"Git Commit: {git_commit} (Dirty: {is_dirty})")
    print(f"Total number of trials: {executed_trials}")
    print(
        f"Code generation success rate / Average reward / Task completed: \n{success_rate:.3f}/{average_reward:.3f}/{task_completed_count}"
    )
    print(f"Average code blocks: {average_code_blocks:.3f}")
    print(f"Average regenerations: {average_regenerations:.3f}")
    print(f"Average finishes: {average_finishes:.3f}")
    print(
        f"Token usage (all trials): input={total_input_tokens} "
        f"output={total_output_tokens} reasoning={total_reasoning_tokens} "
        f"cached={total_cached_tokens} calls={total_token_calls}"
    )
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Write summaries to a txt file
    if config["output_dir"]:
        with open(Path(config["output_dir"]) / "summaries.txt", "w") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Visual Differencing Model: {args.visual_differencing_model}\n") if config[
                "use_img_differencing"
            ] else ""
            f.write(f"Config Path: {args.config_path}\n")
            f.write(f"Git Commit: {git_commit} (Dirty: {is_dirty})\n")
            f.write(f"Total number of trials: {executed_trials}\n")
            f.write(
                f"Code generation success rate / Average reward / Task completed: \n{success_rate:.3f}/{average_reward:.3f}/{task_completed_count}\n"
            )
            f.write(f"Average code blocks: {average_code_blocks:.3f}\n")
            f.write(f"Average regenerations: {average_regenerations:.3f}\n")
            f.write(f"Average finishes: {average_finishes:.3f}\n")
            f.write(
                f"Token usage (all trials): input={total_input_tokens} "
                f"output={total_output_tokens} "
                f"reasoning={total_reasoning_tokens} "
                f"cached={total_cached_tokens} "
                f"calls={total_token_calls}\n"
            )
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
