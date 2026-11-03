import argparse
import copy
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from common.seed_utils import set_all_seeds
from common.logging_utils import save_json, load_json
from task1.data_prep import (
    select_metamath_40k,
    sample_activation_probe_ids,
    apply_chat_template_to_ids,
)
from task1.adapters import (
    inject_adapters,
    compute_top_k_svd_bases,
    merge_adapters_into_backbone,
    verify_merged_vs_unmerged_logits,
    LoRALinear,
)
from task1.trainer import train_task1_variant
from task1.evaluate_harness import run_lm_eval_for_condition, compute_benchmark_deltas_table
from task1.diagnostics import (
    DIAGNOSTIC_LAYERS,
    build_24_qualitative_prompts,
    get_down_proj_module,
    extract_effective_down_proj_delta_w,
    compute_weight_space_diagnostics,
    cache_starting_down_proj_activations,
    compute_activation_perturbation_ratios,
    generate_qualitative_suite,
)
from task1.plots_task1 import plot_task1_learning_curves, plot_task1_diagnostics

MODEL_ID = "Qwen/Qwen3-0.6B"


def load_qwen_backbone(dtype: torch.dtype, device: str = "cuda"):
    """Load Qwen3-0.6B cleanly across both transformers 4.x and 5.x."""
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    return model.to(dtype=dtype, device=device)


def load_trained_condition_model(condition: str, out_dir: Path, is_smoke: bool = False):
    """Load a trained checkpoint or adapter state into a fresh Qwen3-0.6B backbone[cite: 3, 4]."""
    final_u = 2 if is_smoke else 1250
    mid_u = 1 if is_smoke else 625

    if condition == "baseline":
        return load_qwen_backbone(torch.float16, device="cuda")

    if condition in ("full_ft_mid", "full_ft", "l2_sp"):
        var_folder = "full_ft" if condition.startswith("full_ft") else "l2_sp"
        u_num = mid_u if condition == "full_ft_mid" else final_u
        ckpt_path = out_dir / var_folder / "checkpoints" / f"checkpoint_update_{u_num}.pt"
        model = load_qwen_backbone(torch.float32, device="cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(dtype=torch.float16, device="cuda")

    model = load_qwen_backbone(torch.float16, device="cuda")
    r = 4 if condition == "lora_4" else 16
    alpha = 8.0 if condition == "lora_4" else 32.0
    use_oplora = condition == "oplora_16"
    inject_adapters(
        model, r=r, alpha=alpha, dropout=0.05, use_oplora=use_oplora, protection_rank_k=128, init_seed=6304
    )
    adapter_path = out_dir / condition / "checkpoints" / f"adapter_update_{final_u}.pt"
    state = torch.load(adapter_path, map_location="cuda")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(state[f"{name}.lora_A"].to(module.lora_A.device))
            module.lora_B.data.copy_(state[f"{name}.lora_B"].to(module.lora_B.device))
    return model


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 1 Runner")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/task1_outputs")
    parser.add_argument("--stage", type=str, choices=["smoke", "prep", "train", "eval_and_diag"], default="smoke")
    parser.add_argument("--variant", type=str, choices=["full_ft", "l2_sp", "lora_4", "lora_16", "oplora_16", "all"], default="all")
    parser.add_argument("--yaml-path", type=str, default="first_plot_questions.yaml")
    parser.add_argument("--prune-optimizer-states", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(6304)

    is_smoke = args.stage == "smoke"
    target_examples = 64 if is_smoke else 40000
    total_updates = 2 if is_smoke else 1250
    warmup_updates = 1 if is_smoke else 38
    recovery_updates = (1, 2) if is_smoke else (312, 625, 937, 1250)
    eval_limit = 8 if is_smoke else None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_cache_path = out_dir / ("smoke_data.pt" if is_smoke else "metamath_40k.pt")
    probe_path = out_dir / "probe_ids.json"

    if args.stage in ("smoke", "prep") or not data_cache_path.exists():
        gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
        gsm8k_questions = [ex["question"] for ex in gsm8k_ds]
        mm_ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=is_smoke)
        retained, stats = select_metamath_40k(
            mm_ds, gsm8k_questions, tokenizer, target_count=target_examples, max_seq_len=2048
        )
        print(f"[Data Prep] Retained {len(retained)} examples. Stats: {stats}")
        torch.save(retained, data_cache_path)
        save_json(stats, out_dir / "data_selection_stats.json")
        sample_activation_probe_ids(
            num_gsm8k=len(gsm8k_questions), num_hellaswag=10042, sample_size=32, seed=6304, save_path=probe_path
        )
        if args.stage == "prep":
            return

    examples = torch.load(data_cache_path)
    variants_to_run = (
        ["full_ft", "l2_sp", "lora_4", "lora_16", "oplora_16"]
        if args.variant == "all"
        else [args.variant]
    )

    if args.stage in ("smoke", "train"):
        histories = {}
        for var in variants_to_run:
            print(f"\n--- Training Variant: {var} ({total_updates} updates) ---")
            set_all_seeds(6304)
            if var in ("full_ft", "l2_sp"):
                model = load_qwen_backbone(torch.float32, device="cuda")
                res = train_task1_variant(
                    model=model,
                    examples=examples,
                    variant_name=var,
                    output_dir=out_dir,
                    peak_lr=2e-5,
                    l2_sp_lambda=1e-3 if var == "l2_sp" else 0.0,
                    total_updates=total_updates,
                    warmup_updates=warmup_updates,
                    pad_token_id=tokenizer.pad_token_id,
                    is_adapter_run=False,
                    track_displacement=True,
                    recovery_updates=recovery_updates,
                )
            else:
                model = load_qwen_backbone(torch.float16, device="cuda")
                r = 4 if var == "lora_4" else 16
                alpha = 8.0 if var == "lora_4" else 32.0
                use_oplora = var == "oplora_16"
                meta = inject_adapters(
                    model, r=r, alpha=alpha, dropout=0.05, use_oplora=use_oplora, protection_rank_k=128, init_seed=6304
                )
                res = train_task1_variant(
                    model=model,
                    examples=examples,
                    variant_name=var,
                    output_dir=out_dir,
                    peak_lr=2e-4,
                    l2_sp_lambda=0.0,
                    total_updates=total_updates,
                    warmup_updates=warmup_updates,
                    pad_token_id=tokenizer.pad_token_id,
                    is_adapter_run=True,
                    track_displacement=False,
                    recovery_updates=recovery_updates,
                    adapter_meta=meta,
                )
                mb_ids = torch.tensor([examples[0]["input_ids"][:64]], dtype=torch.long, device="cuda")
                merged_copy = merge_adapters_into_backbone(copy.deepcopy(model))
                logit_diff = verify_merged_vs_unmerged_logits(model, merged_copy, mb_ids)
                save_json(logit_diff, out_dir / var / "merged_logit_verification.json")
                del merged_copy

            if args.prune_optimizer_states:
                for u_drop in (312, 937):
                    p_drop = out_dir / var / "checkpoints" / f"checkpoint_update_{u_drop}.pt"
                    if p_drop.exists():
                        p_drop.unlink()

            histories[var] = res["history"]
            del model
            torch.cuda.empty_cache()

        plot_task1_learning_curves(histories, out_dir / "figures" / "task1_learning_curves.pdf")

    if args.stage in ("smoke", "eval_and_diag"):
        base_model = load_qwen_backbone(torch.float16, device="cuda")
        W0_dict = {
            l: get_down_proj_module(base_model, l).weight.detach().clone().float().cpu()
            for l in DIAGNOSTIC_LAYERS
        }
        Uk_dict = {
            l: compute_top_k_svd_bases(W0_dict[l], k=128)[0].cpu()
            for l in DIAGNOSTIC_LAYERS
        }

        probe_info = load_json(probe_path)
        gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
        hswag_ds = load_dataset("Rowan/hellaswag", split="validation")

        math_prompts = [
            apply_chat_template_to_ids(
                tokenizer,
                [{"role": "user", "content": gsm8k_ds[i]["question"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for i in probe_info["gsm8k_test_ids"][: (4 if is_smoke else 32)]
        ]
        nonmath_prompts = [
            tokenizer.encode(hswag_ds[i]["ctx"], add_special_tokens=False)
            for i in probe_info["hellaswag_val_ids"][: (4 if is_smoke else 32)]
        ]

        Z_math = cache_starting_down_proj_activations(base_model, math_prompts, device="cuda")
        Z_nonmath = cache_starting_down_proj_activations(base_model, nonmath_prompts, device="cuda")
        del base_model
        torch.cuda.empty_cache()

        prompts_24 = build_24_qualitative_prompts(args.yaml_path)
        if is_smoke:
            prompts_24 = [p for p in prompts_24 if p["question_index"] in (0, 7)]

        all_conditions = [
            "baseline",
            "full_ft_mid",
            "full_ft",
            "l2_sp",
            "lora_4",
            "lora_16",
            "oplora_16",
        ]
        summaries = []
        qual_all, qual_42 = [], []
        weight_diag, act_diag = {}, {}

        for cond in all_conditions:
            print(f"\n--- Evaluating & Diagnosing Condition: {cond} ---")
            model = load_trained_condition_model(cond, out_dir, is_smoke=is_smoke)

            if cond != "baseline":
                weight_diag[cond] = {}
                act_diag[cond] = {}
                for l in DIAGNOSTIC_LAYERS:
                    mod_l = get_down_proj_module(model, l)
                    dW = extract_effective_down_proj_delta_w(mod_l, W0_dict[l]).cpu()
                    weight_diag[cond][l] = compute_weight_space_diagnostics(
                        W0_dict[l], dW, U_k_fp32=Uk_dict[l], k=128
                    )
                    act_diag[cond][l] = {
                        "math": compute_activation_perturbation_ratios(W0_dict[l], dW, Z_math[l]),
                        "non_math": compute_activation_perturbation_ratios(W0_dict[l], dW, Z_nonmath[l]),
                    }

            if cond in ("lora_4", "lora_16", "oplora_16"):
                model = merge_adapters_into_backbone(model)

            s = run_lm_eval_for_condition(
                model, tokenizer, cond, out_dir / "eval", batch_size=8, limit=eval_limit
            )
            summaries.append(s)

            q_out = generate_qualitative_suite(
                model, tokenizer, cond, prompts_24, max_new_tokens=(64 if is_smoke else 512)
            )
            qual_all.extend(q_out["all_24"])
            qual_42.extend(q_out["annotated_first_last_6"])

            del model
            torch.cuda.empty_cache()

        deltas_table = compute_benchmark_deltas_table(summaries, baseline_condition="baseline")
        save_json(summaries, out_dir / "eval" / "all_benchmark_summaries.json")
        save_json(deltas_table, out_dir / "eval" / "benchmark_deltas_table.json")
        save_json(weight_diag, out_dir / "eval" / "weight_space_diagnostics.json")
        save_json(act_diag, out_dir / "eval" / "activation_diagnostics.json")
        save_json(qual_all, out_dir / "eval" / "qualitative_168.json")
        save_json(qual_42, out_dir / "eval" / "qualitative_42_subset.json")

        plot_task1_diagnostics(
            weight_diag, act_diag, out_dir / "figures" / "task1_diagnostics.pdf"
        )


if __name__ == "__main__":
    main()
