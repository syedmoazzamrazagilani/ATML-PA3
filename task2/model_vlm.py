from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from task1.adapters import inject_adapters, LoRALinear

TASK2_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class VisualConnector(nn.Module):
    """
    Two-layer MLP mapping cached SigLIP patch features F(I) in R^{196 x 768}
    (or R^{49 x 768} when pool_2x2=True) into Qwen's embedding space R^{M x 1024}.
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        pool_2x2: bool = False,
    ):
        super().__init__()
        self.pool_2x2 = pool_2x2
        self.num_visual_tokens = 49 if pool_2x2 else 196
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.calibration_scale = 1.0
        self.calibration_rule = (
            "Rescale fc2.weight and fc2.bias by RMS(E_text_train) / RMS(g_phi(F_train)) "
            "using training image features and training caption token embeddings only."
        )

    def forward(self, vis_feats: torch.Tensor) -> torch.Tensor:
        x = vis_feats.to(torch.float32)
        if self.pool_2x2 and x.size(1) == 196:
            B, _, D = x.shape
            x = (
                x.view(B, 14, 14, D)
                .view(B, 7, 2, 7, 2, D)
                .mean(dim=(2, 4))
                .reshape(B, 49, D)
            )
        return self.fc2(self.act(self.fc1(x)))

    @torch.no_grad()
    def calibrate_to_text_embeddings(
        self, sample_train_vis_feats: torch.Tensor, text_embed_weight: torch.Tensor
    ) -> Dict[str, Any]:
        """Calibrate connector initial output RMS magnitude against Qwen text embeddings using training data only."""
        device = next(self.parameters()).device
        vis_out = self.forward(sample_train_vis_feats.to(device))
        vis_rms = float(torch.sqrt(torch.mean(vis_out.float() ** 2)).item())
        text_rms = float(
            torch.sqrt(torch.mean(text_embed_weight.detach().float() ** 2)).item()
        )
        scale = text_rms / max(vis_rms, 1e-8)
        self.fc2.weight.data.mul_(scale)
        if self.fc2.bias is not None:
            self.fc2.bias.data.mul_(scale)
        self.calibration_scale = float(scale)
        post_vis_rms = float(
            torch.sqrt(
                torch.mean(self.forward(sample_train_vis_feats.to(device)).float() ** 2)
            ).item()
        )
        return {
            "rule": self.calibration_rule,
            "text_embed_rms": text_rms,
            "pre_calibration_vis_rms": vis_rms,
            "calibration_scale_factor": self.calibration_scale,
            "post_calibration_vis_rms": post_vis_rms,
        }


class SiglipQwenVLM(nn.Module):
    """
    SigLIP-Qwen3-0.6B Vision-Language Model.
    Inserts M in {196, 49} visual tokens at the start of the user message before the question,
    preserving Qwen's non-thinking chat structure.
    """
    def __init__(
        self,
        qwen_model: nn.Module,
        tokenizer: Any,
        pool_2x2: bool = False,
        init_seed: int = 6304,
    ):
        super().__init__()
        self.qwen = qwen_model
        self.tokenizer = tokenizer
        self.pool_2x2 = pool_2x2
        self.init_seed = init_seed

        # Freeze all Qwen backbone parameters in FP16
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

        torch.manual_seed(init_seed)
        device = next(self.qwen.parameters()).device
        self.connector = VisualConnector(
            768, 1024, 1024, pool_2x2=pool_2x2
        ).to(device=device, dtype=torch.float32)
        self.lora_enabled = False

        # Precompute chat template boundary token IDs (with thinking disabled)
        self.user_prefix_ids = self.tokenizer.encode(
            "<|im_start|>user\n", add_special_tokens=False
        )
        self.assistant_prefix_ids = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.im_end_id = (
            int(im_end)
            if isinstance(im_end, int) and im_end >= 0
            else int(self.tokenizer.eos_token_id)
        )
        self.pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.im_end_id
        )

    def enable_stage2_lora(self) -> Dict[str, float]:
        """Enable LoRA (r=16, alpha=32, dropout=0.05) on q_proj, k_proj, v_proj, o_proj for Stages 2-3."""
        if self.lora_enabled:
            return {"trainable_params": self.count_trainable_params()["total"]}
        meta = inject_adapters(
            self.qwen,
            r=16,
            alpha=32.0,
            dropout=0.05,
            use_oplora=False,
            target_modules=TASK2_LORA_TARGETS,
            init_seed=self.init_seed,
        )
        # Ensure connector remains trainable in FP32
        for p in self.connector.parameters():
            p.requires_grad_(True)
        self.lora_enabled = True
        return meta

    def count_trainable_params(self) -> Dict[str, int]:
        conn_p = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
        lora_p = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        return {"connector": conn_p, "lora": lora_p, "total": conn_p + lora_p}

    def build_tokenized_text_parts(
        self,
        question: str,
        target: Optional[str] = None,
        is_caption: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Apply Task 2 content-token caps:
          - Question: 80 tokens
          - Caption target: 64 tokens
          - Short-answer target: 16 tokens
        """
        q_ids = self.tokenizer.encode(question.strip(), add_special_tokens=False)[:80]
        after_vis_ids = q_ids + self.assistant_prefix_ids

        target_content_ids: List[int] = []
        if target is not None:
            cap = 64 if is_caption else 16
            target_content_ids = self.tokenizer.encode(
                target.strip(), add_special_tokens=False
            )[:cap]

        return {
            "prefix_ids": list(self.user_prefix_ids),
            "after_vis_ids": after_vis_ids,
            "target_content_ids": target_content_ids,
            "target_with_eot_ids": target_content_ids + [self.im_end_id]
            if target is not None
            else [],
        }

    def prepare_multimodal_batch(
        self,
        batch_items: List[Dict[str, Any]],
        vis_feats_batch: Optional[torch.Tensor],
        has_image: bool = True,
        exclude_eot_from_labels: bool = False,
    ) -> Dict[str, Any]:
        """
        Splice continuous visual embeddings into the user message before the question:
          [Embed(<|im_start|>user\n), Z(I), Embed(question + assistant_prefix), Embed(target + <|im_end|>)]
        Constructs matching labels (-100 on visual tokens, prompts, and padding),
        attention_mask, position_ids, and per-example valid token counts |R_i|.
        If has_image=False (No-image condition), omits visual tokens and rebuilds masks and positions.
        """
        device = next(self.qwen.parameters()).device
        embed_layer = self.qwen.model.embed_tokens
        embed_dtype = embed_layer.weight.dtype

        if has_image and vis_feats_batch is not None:
            vis_embeds = self.connector(vis_feats_batch.to(device)).to(embed_dtype)
            num_vis = vis_embeds.size(1)
        else:
            vis_embeds = None
            num_vis = 0

        seq_embeds_list = []
        labels_list = []
        valid_counts = []

        for idx, item in enumerate(batch_items):
            parts = self.build_tokenized_text_parts(
                question=item["question"],
                target=item.get("target"),
                is_caption=item.get("is_caption", False),
            )
            p_ids = torch.tensor(parts["prefix_ids"], dtype=torch.long, device=device)
            av_ids = torch.tensor(parts["after_vis_ids"], dtype=torch.long, device=device)

            t_full_ids = parts["target_with_eot_ids"]
            t_tensor = torch.tensor(t_full_ids, dtype=torch.long, device=device)

            p_emb = embed_layer(p_ids)
            av_emb = embed_layer(av_ids)
            t_emb = embed_layer(t_tensor) if len(t_full_ids) > 0 else p_emb[:0]

            if has_image and vis_embeds is not None:
                v_emb = vis_embeds[idx]
                full_emb = torch.cat([p_emb, v_emb, av_emb, t_emb], dim=0)
                prompt_len = p_ids.numel() + num_vis + av_ids.numel()
            else:
                full_emb = torch.cat([p_emb, av_emb, t_emb], dim=0)
                prompt_len = p_ids.numel() + av_ids.numel()

            if exclude_eot_from_labels and len(t_full_ids) > 0:
                # For reference-answer NLL, exclude termination token (<|im_end|>)
                sup_ids = parts["target_content_ids"] + [-100]
            else:
                sup_ids = list(t_full_ids)

            lbl = [-100] * prompt_len + sup_ids
            r_i = sum(1 for x in sup_ids if x != -100)
            seq_embeds_list.append(full_emb)
            labels_list.append(torch.tensor(lbl, dtype=torch.long, device=device))
            valid_counts.append(max(1, r_i))

        max_len = max(e.size(0) for e in seq_embeds_list)
        pad_emb = embed_layer(
            torch.tensor([self.pad_token_id], dtype=torch.long, device=device)
        )

        padded_embeds = []
        padded_labels = []
        attn_masks = []
        pos_ids = []

        for emb, lbl in zip(seq_embeds_list, labels_list):
            cur_len = emb.size(0)
            pad_len = max_len - cur_len
            if pad_len > 0:
                padded_embeds.append(
                    torch.cat([emb, pad_emb.expand(pad_len, -1)], dim=0)
                )
                padded_labels.append(
                    torch.cat(
                        [
                            lbl,
                            torch.full(
                                (pad_len,), -100, dtype=torch.long, device=device
                            ),
                        ],
                        dim=0,
                    )
                )
                attn_masks.append(
                    torch.cat(
                        [
                            torch.ones(cur_len, dtype=torch.long, device=device),
                            torch.zeros(pad_len, dtype=torch.long, device=device),
                        ],
                        dim=0,
                    )
                )
            else:
                padded_embeds.append(emb)
                padded_labels.append(lbl)
                attn_masks.append(torch.ones(cur_len, dtype=torch.long, device=device))
            pos_ids.append(torch.arange(max_len, dtype=torch.long, device=device))

        return {
            "inputs_embeds": torch.stack(padded_embeds, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(attn_masks, dim=0),
            "position_ids": torch.stack(pos_ids, dim=0),
            "valid_counts": torch.tensor(valid_counts, dtype=torch.float32, device=device),
        }

    def compute_per_example_losses(
        self,
        batch_items: List[Dict[str, Any]],
        vis_feats_batch: Optional[torch.Tensor],
        has_image: bool = True,
        exclude_eot_from_labels: bool = False,
    ) -> torch.Tensor:
        """
        Compute per-example token-averaged negative log-likelihood:
            l_i = - (1 / |R_i|) * sum_{t in R_i} log p(y_{i,t} | I_i, q_i, y_{i,<t})
        Returns tensor of shape (B,) in FP32.
        """
        prepared = self.prepare_multimodal_batch(
            batch_items,
            vis_feats_batch,
            has_image=has_image,
            exclude_eot_from_labels=exclude_eot_from_labels,
        )
        inputs_embeds = prepared["inputs_embeds"]
        labels = prepared["labels"]
        attention_mask = prepared["attention_mask"]
        position_ids = prepared["position_ids"]
        valid_counts = prepared["valid_counts"]

        backbone_out = self.qwen.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = (
            backbone_out.last_hidden_state
            if hasattr(backbone_out, "last_hidden_state")
            else backbone_out[0]
        )

        shift_h = hidden_states[..., :-1, :]
        shift_labels = labels[..., 1:]
        B, S_minus_1, _ = shift_h.shape

        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return hidden_states.sum() * 0.0

        valid_h = shift_h[valid_mask]
        valid_targets = shift_labels[valid_mask]
        batch_indices = (
            torch.arange(B, device=hidden_states.device)
            .unsqueeze(1)
            .expand(B, S_minus_1)[valid_mask]
        )

        lm_head = self.qwen.lm_head
        logits_valid = F.linear(valid_h, lm_head.weight, lm_head.bias).float()
        token_losses = F.cross_entropy(logits_valid, valid_targets, reduction="none")

        per_ex_sum = torch.zeros(B, device=hidden_states.device, dtype=torch.float32)
        per_ex_sum.index_add_(0, batch_indices, token_losses)
        return per_ex_sum / valid_counts

    @torch.no_grad()
    def greedy_generate_batch(
        self,
        questions: List[str],
        vis_feats_batch: Optional[torch.Tensor],
        has_image: bool = True,
        max_new_tokens: int = 16,
    ) -> List[str]:
        """
        Greedy decoding with thinking disabled and a 16-new-token cap across a batch of prompts.
        Left-pads prompt embeddings so generation aligns cleanly at the right edge.
        """
        self.eval()
        device = next(self.qwen.parameters()).device
        embed_layer = self.qwen.model.embed_tokens
        embed_dtype = embed_layer.weight.dtype

        if has_image and vis_feats_batch is not None:
            vis_embeds = self.connector(vis_feats_batch.to(device)).to(embed_dtype)
        else:
            vis_embeds = None

        prompt_embeds_list = []
        for idx, q in enumerate(questions):
            parts = self.build_tokenized_text_parts(q, target=None, is_caption=False)
            p_emb = embed_layer(
                torch.tensor(parts["prefix_ids"], dtype=torch.long, device=device)
            )
            av_emb = embed_layer(
                torch.tensor(parts["after_vis_ids"], dtype=torch.long, device=device)
            )
            if has_image and vis_embeds is not None:
                full_p = torch.cat([p_emb, vis_embeds[idx], av_emb], dim=0)
            else:
                full_p = torch.cat([p_emb, av_emb], dim=0)
            prompt_embeds_list.append(full_p)

        B = len(prompt_embeds_list)
        max_p_len = max(e.size(0) for e in prompt_embeds_list)
        pad_emb = embed_layer(
            torch.tensor([self.pad_token_id], dtype=torch.long, device=device)
        )

        # Left-pad prompts for batched autoregressive greedy decoding
        cur_embeds = []
        attn_mask = []
        for emb in prompt_embeds_list:
            pad_len = max_p_len - emb.size(0)
            if pad_len > 0:
                cur_embeds.append(torch.cat([pad_emb.expand(pad_len, -1), emb], dim=0))
                attn_mask.append(
                    torch.cat(
                        [
                            torch.zeros(pad_len, dtype=torch.long, device=device),
                            torch.ones(emb.size(0), dtype=torch.long, device=device),
                        ],
                        dim=0,
                    )
                )
            else:
                cur_embeds.append(emb)
                attn_mask.append(torch.ones(emb.size(0), dtype=torch.long, device=device))

        inputs_embeds = torch.stack(cur_embeds, dim=0)
        attention_mask = torch.stack(attn_mask, dim=0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated_tokens: List[List[int]] = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            pos_ids = attention_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(attention_mask == 0, 0)
            out = self.qwen.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                use_cache=False,
            )
            last_h = (
                out.last_hidden_state[:, -1, :]
                if hasattr(out, "last_hidden_state")
                else out[0][:, -1, :]
            )
            logits = self.qwen.lm_head(last_h)
            next_ids = torch.argmax(logits, dim=-1)

            for b_idx in range(B):
                if not finished[b_idx]:
                    tok_id = int(next_ids[b_idx].item())
                    if tok_id in (self.im_end_id, self.tokenizer.eos_token_id):
                        finished[b_idx] = True
                    else:
                        generated_tokens[b_idx].append(tok_id)

            if finished.all():
                break

            next_emb = embed_layer(next_ids).unsqueeze(1)
            inputs_embeds = torch.cat([inputs_embeds, next_emb], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((B, 1), dtype=torch.long, device=device),
                ],
                dim=1,
            )

        return [
            self.tokenizer.decode(toks, skip_special_tokens=True).strip()
            for toks in generated_tokens
        ]
