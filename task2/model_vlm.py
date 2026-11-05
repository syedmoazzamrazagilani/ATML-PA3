import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from task1.adapters import inject_adapters, LoRALinear

TASK2_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class VisualConnectorMLP(nn.Module):
    """
    Trainable 2-layer MLP connector: Linear(768 -> 1024) -> GELU -> Linear(1024 -> 1024).
    Kept in FP32 for master parameter precision.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 1024, seed: int = 6304):
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5), generator=gen)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5), generator=gen)
        nn.init.zeros_(self.fc2.bias)
        self.calibration_scale = 1.0

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        x = visual_feats.to(torch.float32)
        return self.fc2(self.act(self.fc1(x)))

    @torch.no_grad()
    def calibrate_to_text_embeddings(
        self,
        sample_visual_feats: torch.Tensor,
        text_embed_weight: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Calibrate connector output RMS magnitude against Qwen text embedding RMS magnitude
        using training data only, rescaling fc2 weight and bias in-place.
        """
        device = next(self.parameters()).device
        vis_out = self.forward(sample_visual_feats.to(device))
        vis_rms = float(torch.sqrt(torch.mean(vis_out ** 2)).item())
        txt_rms = float(torch.sqrt(torch.mean(text_embed_weight.detach().float() ** 2)).item())
        scale = txt_rms / max(vis_rms, 1e-8)
        self.fc2.weight.data.mul_(scale)
        self.fc2.bias.data.mul_(scale)
        self.calibration_scale = scale
        post_out = self.forward(sample_visual_feats.to(device))
        post_rms = float(torch.sqrt(torch.mean(post_out ** 2)).item())
        return {
            "rule": "rms_match_fc2_rescale_on_train_features",
            "pre_calibration_vis_rms": vis_rms,
            "target_text_embed_rms": txt_rms,
            "calibration_scale_factor": scale,
            "post_calibration_vis_rms": post_rms,
        }


class SiglipQwenVLM(nn.Module):
    """
    SigLIP-Qwen3-0.6B Vision-Language Model.
    Inserts continuous visual embeddings Z(I) in R^{N_vis x 1024} into the user message
    immediately before the question tokens, preserving Qwen's non-thinking chat structure.
    """
    def __init__(
        self,
        qwen_model: nn.Module,
        num_visual_tokens: int = 196,
        seed: int = 6304,
    ):
        super().__init__()
        self.qwen = qwen_model
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        if hasattr(self.qwen, "config"):
            self.qwen.config.use_cache = False
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            try:
                self.qwen.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.qwen.gradient_checkpointing_enable()
        if hasattr(self.qwen, "enable_input_require_grads"):
            self.qwen.enable_input_require_grads()

        d_model = self.qwen.get_input_embeddings().weight.shape[1]
        self.connector = VisualConnectorMLP(
            in_dim=768, hidden_dim=d_model, out_dim=d_model, seed=seed
        ).to(device=next(self.qwen.parameters()).device, dtype=torch.float32)

        self.num_visual_tokens = num_visual_tokens
        self.lora_enabled = False
        self.calibration_info: Dict[str, Any] = {}

    def calibrate_connector(self, sample_train_feats: torch.Tensor) -> Dict[str, Any]:
        txt_w = self.qwen.get_input_embeddings().weight
        self.calibration_info = self.connector.calibrate_to_text_embeddings(
            sample_train_feats, txt_w
        )
        return self.calibration_info

    def enable_stage2_lora(
        self, r: int = 16, alpha: float = 32.0, dropout: float = 0.05, init_seed: int = 6304
    ) -> Dict[str, float]:
        """Enable LoRA adapters on q_proj, k_proj, v_proj, o_proj for Stages 2 and 3."""
        if self.lora_enabled:
            return self.get_parameter_breakdown()
        inject_adapters(
            self.qwen,
            r=r,
            alpha=alpha,
            dropout=dropout,
            use_oplora=False,
            target_modules=TASK2_LORA_TARGETS,
            init_seed=init_seed,
        )
        # Ensure connector remains trainable
        for p in self.connector.parameters():
            p.requires_grad_(True)
        self.lora_enabled = True
        return self.get_parameter_breakdown()

    def get_parameter_breakdown(self) -> Dict[str, int]:
        conn_params = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        return {
            "connector_trainable_params": conn_params,
            "lora_trainable_params": lora_params,
            "total_trainable_params": conn_params + lora_params,
        }

    def get_trainable_state_dict(self) -> Dict[str, Any]:
        state = {
            "connector": {k: v.detach().cpu() for k, v in self.connector.state_dict().items()},
            "lora_enabled": self.lora_enabled,
            "lora": {},
        }
        if self.lora_enabled:
            for name, mod in self.qwen.named_modules():
                if isinstance(mod, LoRALinear):
                    state["lora"][f"{name}.lora_A"] = mod.lora_A.detach().cpu()
                    state["lora"][f"{name}.lora_B"] = mod.lora_B.detach().cpu()
        return state

    def load_trainable_state_dict(self, state: Dict[str, Any]) -> None:
        self.connector.load_state_dict(state["connector"])
        if state.get("lora_enabled", False):
            if not self.lora_enabled:
                self.enable_stage2_lora()
            lora_dict = state.get("lora", {})
            for name, mod in self.qwen.named_modules():
                if isinstance(mod, LoRALinear) and f"{name}.lora_A" in lora_dict:
                    mod.lora_A.data.copy_(lora_dict[f"{name}.lora_A"].to(mod.lora_A.device))
                    mod.lora_B.data.copy_(lora_dict[f"{name}.lora_B"].to(mod.lora_B.device))

    def _tokenize_segments(
        self,
        tokenizer: Any,
        question: str,
        target_text: Optional[str],
        is_caption: bool,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Construct token segments preserving Qwen3's non-thinking chat structure:
          prefix_ids: `<|im_start|>user\n`
          middle_ids: `<question_tokens><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
          target_ids: `<target_tokens><|im_end|>`
        Enforces content-token caps: question=80, caption=64, short-answer=16.
        """
        prefix_ids = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        q_ids = tokenizer.encode(question.strip(), add_special_tokens=False)[:80]
        suffix_prompt_ids = tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        middle_ids = q_ids + suffix_prompt_ids

        if target_text is None:
            return prefix_ids, middle_ids, []

        cap_limit = 64 if is_caption else 16
        t_ids = tokenizer.encode(target_text.strip(), add_special_tokens=False)[:cap_limit]
        im_end_id = getattr(tokenizer, "eos_token_id", 151645)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                cand = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if isinstance(cand, int) and cand >= 0:
                    im_end_id = cand
            except Exception:
                pass
        target_ids = t_ids + [im_end_id]
        return prefix_ids, middle_ids, target_ids

    def _build_multimodal_batch(
        self,
        batch_items: List[Dict[str, Any]],
        tokenizer: Any,
        feature_store: Any,
        condition: str = "correct",
        for_generation: bool = False,
        include_eot_in_labels: bool = True,
    ) -> Dict[str, Any]:
        """
        Splice continuous visual embeddings into the user turn before the question tokens:
          [prefix_embeds ("<|im_start|>user\n"), visual_embeds (N_vis), middle_embeds (question + assistant header), target_embeds]
        For `condition == "no_image"`, omits visual tokens and rebuilds attention_mask, labels, and position_ids.
        Uses left-padding for generation batches and right-padding for teacher-forced training/NLL batches.
        """
        device = next(self.qwen.parameters()).device
        embed_layer = self.qwen.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype

        use_visual = condition != "no_image"
        if use_visual:
            img_ids = []
            for item in batch_items:
                if condition == "correct":
                    img_ids.append(item["image_id"])
                elif condition == "complementary":
                    img_ids.append(item["partner_image_id"])
                elif condition == "mismatched":
                    img_ids.append(item["mismatched_image_id"])
                else:
                    img_ids.append(item["image_id"])
            raw_vis_feats = feature_store.get_batch_features(img_ids, device=device)
            vis_embeds_batch = self.connector(raw_vis_feats).to(dtype=embed_dtype)
            n_vis = vis_embeds_batch.size(1)
        else:
            vis_embeds_batch = None
            n_vis = 0

        seq_embeds_list = []
        seq_labels_list = []
        for idx, item in enumerate(batch_items):
            prefix_ids, middle_ids, target_ids = self._tokenize_segments(
                tokenizer,
                item["question"],
                None if for_generation else item["target_text"],
                item.get("is_caption", False),
            )
            p_emb = embed_layer(torch.tensor(prefix_ids, dtype=torch.long, device=device))
            m_emb = embed_layer(torch.tensor(middle_ids, dtype=torch.long, device=device))

            parts = [p_emb]
            if use_visual and vis_embeds_batch is not None:
                parts.append(vis_embeds_batch[idx])
            parts.append(m_emb)

            prompt_total_len = len(prefix_ids) + n_vis + len(middle_ids)
            if not for_generation:
                t_emb = embed_layer(torch.tensor(target_ids, dtype=torch.long, device=device))
                parts.append(t_emb)
                labels_i = [-100] * prompt_total_len + list(target_ids)
                if not include_eot_in_labels and len(labels_i) > prompt_total_len:
                    # Mask termination token (<|im_end|>) for answer NLL evaluation
                    labels_i[-1] = -100
                seq_labels_list.append(labels_i)

            seq_embeds_list.append(torch.cat(parts, dim=0))

        max_len = max(x.size(0) for x in seq_embeds_list)
        d_model = seq_embeds_list[0].size(1)
        B = len(seq_embeds_list)

        padded_embeds = torch.zeros((B, max_len, d_model), dtype=embed_dtype, device=device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        position_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
        padded_labels = (
            torch.full((B, max_len), -100, dtype=torch.long, device=device)
            if not for_generation
            else None
        )

        for i, emb in enumerate(seq_embeds_list):
            L = emb.size(0)
            if for_generation:
                # Left-pad for batched autoregressive generation
                pad_len = max_len - L
                padded_embeds[i, pad_len:, :] = emb
                attention_mask[i, pad_len:] = 1
                position_ids[i, pad_len:] = torch.arange(L, dtype=torch.long, device=device)
            else:
                padded_embeds[i, :L, :] = emb
                attention_mask[i, :L] = 1
                position_ids[i, :L] = torch.arange(L, dtype=torch.long, device=device)
                padded_labels[i, :L] = torch.tensor(seq_labels_list[i], dtype=torch.long, device=device)

        return {
            "inputs_embeds": padded_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": padded_labels,
        }

    def forward_example_weighted_loss_sum(
        self,
        batch_items: List[Dict[str, Any]],
        tokenizer: Any,
        feature_store: Any,
    ) -> torch.Tensor:
        """
        Compute sum of per-example mean supervised-token losses across the microbatch:
            sum_{i in MB} (1 / |R_i|) sum_{t in R_i} -log p(y_{i,t} | I_i, q_i, y_{i,<t})
        Projects only valid supervised positions through `lm_head` to avoid full-vocabulary VRAM spikes.
        """
        batch = self._build_multimodal_batch(
            batch_items, tokenizer, feature_store, condition="correct", for_generation=False, include_eot_in_labels=True
        )
        backbone_out = self.qwen.model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        )
        hidden = (
            backbone_out.last_hidden_state
            if hasattr(backbone_out, "last_hidden_state")
            else backbone_out[0]
        )
        shift_h = hidden[:, :-1, :]
        shift_labels = batch["labels"][:, 1:]
        valid_mask = shift_labels != -100

        valid_h = shift_h[valid_mask]
        valid_targets = shift_labels[valid_mask]
        if valid_targets.numel() == 0:
            return hidden.sum() * 0.0

        # Compute token weights 1 / |R_i| for each valid token so summing equals sum_i (1/|R_i|) sum_t CE_{i,t}
        per_ex_counts = valid_mask.sum(dim=1).clamp(min=1).to(torch.float32)
        ex_weights = (1.0 / per_ex_counts).unsqueeze(1).expand_as(shift_labels)
        valid_weights = ex_weights[valid_mask]

        lm_head = self.qwen.lm_head
        logits_valid = F.linear(valid_h, lm_head.weight, lm_head.bias).float()
        token_ces = F.cross_entropy(logits_valid, valid_targets, reduction="none")
        return torch.sum(token_ces * valid_weights)

    @torch.no_grad()
    def compute_per_example_nll(
        self,
        batch_items: List[Dict[str, Any]],
        tokenizer: Any,
        feature_store: Any,
        include_eot: bool = False,
        condition: str = "correct",
    ) -> List[float]:
        """
        Compute per-example token-averaged negative log-likelihood (NLL).
        For answer NLL (`include_eot=False`), excludes termination token (<|im_end|>).
        """
        batch = self._build_multimodal_batch(
            batch_items,
            tokenizer,
            feature_store,
            condition=condition,
            for_generation=False,
            include_eot_in_labels=include_eot,
        )
        backbone_out = self.qwen.model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        )
        hidden = (
            backbone_out.last_hidden_state
            if hasattr(backbone_out, "last_hidden_state")
            else backbone_out[0]
        )
        shift_h = hidden[:, :-1, :]
        shift_labels = batch["labels"][:, 1:]
        lm_head = self.qwen.lm_head

        nlls = []
        for i in range(len(batch_items)):
            mask_i = shift_labels[i] != -100
            if not torch.any(mask_i):
                nlls.append(0.0)
                continue
            h_i = shift_h[i][mask_i]
            t_i = shift_labels[i][mask_i]
            logits_i = F.linear(h_i, lm_head.weight, lm_head.bias).float()
            nll_i = F.cross_entropy(logits_i, t_i, reduction="mean")
            nlls.append(float(nll_i.item()))
        return nlls

    @torch.no_grad()
    def generate_batch(
        self,
        batch_items: List[Dict[str, Any]],
        tokenizer: Any,
        feature_store: Any,
        condition: str = "correct",
        max_new_tokens: int = 16,
    ) -> List[str]:
        """Greedy batched generation with KV-cache and thinking disabled."""
        batch = self._build_multimodal_batch(
            batch_items, tokenizer, feature_store, condition=condition, for_generation=True
        )
        inputs_embeds = batch["inputs_embeds"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]
        B = inputs_embeds.size(0)
        device = inputs_embeds.device

        im_end_id = getattr(tokenizer, "eos_token_id", 151645)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                cand = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if isinstance(cand, int) and cand >= 0:
                    im_end_id = cand
            except Exception:
                pass

        embed_layer = self.qwen.get_input_embeddings()
        generated_tokens: List[List[int]] = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        past_key_values = None
        cur_embeds = inputs_embeds
        cur_mask = attention_mask
        cur_pos = position_ids

        for _ in range(max_new_tokens):
            outputs = self.qwen(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                position_ids=cur_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            past_key_values = outputs.past_key_values

            for b_idx in range(B):
                if not finished[b_idx]:
                    tok_id = int(next_tokens[b_idx].item())
                    if tok_id in (im_end_id, tokenizer.eos_token_id):
                        finished[b_idx] = True
                    else:
                        generated_tokens[b_idx].append(tok_id)

            if torch.all(finished):
                break

            cur_embeds = embed_layer(next_tokens.unsqueeze(1))
            cur_mask = torch.cat(
                [cur_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1
            )
            cur_pos = cur_pos[:, -1:] + 1

        return [tokenizer.decode(toks, skip_special_tokens=True).strip() for toks in generated_tokens]
