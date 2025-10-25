import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class IFAHead:
    """
    Inference-only IFA/BFP head adapted for arbitrary channel size and backbones.

    This module replicates the core ideas of IFA:
      - Foreground/Background prototype pooling from support masks.
      - SSP refinement to mine confident local regions (global + local prototypes).
      - Iterative update on query side with an optional refine step in the first iteration.

    Differences vs. the original training code:
      - No dependency on a specific backbone; only consumes feature maps [B, C, H, W].
      - Channel dimension C is inferred dynamically (no 1024 hardcoding).
      - Support-side SSP updates are computed in inference for better alignment, but no losses.
    """

    def __init__(
        self,
        temperature: float = 10.0,
        fg_thresh: float = 0.7,
        bg_thresh: float = 0.6,
        topk_fallback: int = 12,
        iters: int = 3,
        use_refine: bool = True,
    ) -> None:
        self.temperature = float(temperature)
        self.fg_thresh = float(fg_thresh)
        self.bg_thresh = float(bg_thresh)
        self.topk_fallback = int(topk_fallback)
        self.iters = int(iters)
        self.use_refine = bool(use_refine)

    @staticmethod
    def _masked_average_pooling(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feature: [B, C, H, W]
        mask:    [B, H, W] in {0,1} (float/long)
        returns: [B, C] pooled features
        """
        # Ensure mask on same device/dtype and shape [B,1,H,W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=feature.device, dtype=feature.dtype)
        # Align mask to feature spatial size
        mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=True)
        masked = feature * mask
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-5)  # [B, 1]
        pooled = masked.sum(dim=(2, 3)) / denom  # [B, C]
        return pooled

    def _similarity_logits(self, feat: torch.Tensor, fg_proto: torch.Tensor, bg_proto: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity to foreground/background prototypes.

        feat:      [B, C, H, W]
        fg_proto:  [B, C, 1, 1] or [1, C, 1, 1]
        bg_proto:  [B, C, 1, 1] or [1, C, 1, 1]
        returns:   [B, 2, H, W] (bg, fg)
        """
        sim_fg = F.cosine_similarity(feat, fg_proto, dim=1)
        sim_bg = F.cosine_similarity(feat, bg_proto, dim=1)
        out = torch.stack([sim_bg, sim_fg], dim=1) * self.temperature
        return out

    def _ssp(
        self,
        feat: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Self-Support Prototype (SSP) mining on a feature map given current logits.

        feat:   [B, C, H, W]
        logits: [B, 2, H, W]
        returns (new_fg, new_bg, new_fg_local, new_bg_local):
            new_fg:       [B, C, 1, 1]
            new_bg:       [B, C, 1, 1]
            new_fg_local: [B, C, H, W]
            new_bg_local: [B, C, H, W]
        """
        b, c, h, w = feat.shape
        pred = logits.softmax(1)  # [B,2,H,W]
        pred_fg = pred[:, 1].reshape(b, -1)  # [B, HW]
        pred_bg = pred[:, 0].reshape(b, -1)  # [B, HW]

        feat_flat = feat.view(b, c, -1)  # [B, C, HW]
        fg_list = []
        bg_list = []
        fg_local_list = []
        bg_local_list = []

        for i in range(b):
            cur = feat_flat[i]  # [C, HW]
            # Foreground selection
            fg_mask = pred_fg[i] > self.fg_thresh
            if torch.count_nonzero(fg_mask) > 0:
                fg_feat = cur[:, fg_mask]
            else:
                k = min(self.topk_fallback, cur.shape[1])
                topk = torch.topk(pred_fg[i], k).indices
                fg_feat = cur[:, topk]
            # Background selection
            bg_mask = pred_bg[i] > self.bg_thresh
            if torch.count_nonzero(bg_mask) > 0:
                bg_feat = cur[:, bg_mask]
            else:
                k = min(self.topk_fallback, cur.shape[1])
                topk = torch.topk(pred_bg[i], k).indices
                bg_feat = cur[:, topk]

            # Global prototypes
            fg_proto = fg_feat.mean(dim=-1)  # [C]
            bg_proto = bg_feat.mean(dim=-1)  # [C]
            fg_list.append(fg_proto.unsqueeze(0))
            bg_list.append(bg_proto.unsqueeze(0))

            # Local prototypes via attention-weighted aggregation
            # Normalize features along channel dim for cosine attention
            fg_feat_n = fg_feat / (fg_feat.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6))  # [C, N1]
            bg_feat_n = bg_feat / (bg_feat.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6))  # [C, N2]
            cur_n = cur / (cur.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6))             # [C, HW]

            # [HW, C] @ [C, N*] -> [HW, N*]
            sim_fg = (cur_n.t() @ fg_feat_n) * 2.0
            sim_bg = (cur_n.t() @ bg_feat_n) * 2.0
            sim_fg = sim_fg.softmax(dim=-1)
            sim_bg = sim_bg.softmax(dim=-1)

            # [HW, N*] @ [N*, C] -> [HW, C]
            fg_local = (sim_fg @ fg_feat.t()).t().view(c, h, w).unsqueeze(0)  # [1, C, H, W]
            bg_local = (sim_bg @ bg_feat.t()).t().view(c, h, w).unsqueeze(0)  # [1, C, H, W]

            fg_local_list.append(fg_local)
            bg_local_list.append(bg_local)

        new_fg = torch.cat(fg_list, dim=0).unsqueeze(-1).unsqueeze(-1)       # [B, C, 1, 1]
        new_bg = torch.cat(bg_list, dim=0).unsqueeze(-1).unsqueeze(-1)       # [B, C, 1, 1]
        new_fg_local = torch.cat(fg_local_list, dim=0)                       # [B, C, H, W]
        new_bg_local = torch.cat(bg_local_list, dim=0)                       # [B, C, H, W]

        return new_fg, new_bg, new_fg_local, new_bg_local

    def _iter_bfp(
        self,
        fp: torch.Tensor,
        bp: torch.Tensor,
        feat_s: Optional[torch.Tensor],  # [K, C, H, W] or None
        feat_q: torch.Tensor,            # [B, C, H, W]
        run_refine: bool,
        shot: int,
    ):
        """
        A single BFP iteration, with optional refine on the query side.
        Returns different tuples depending on refine and training mode. Here, inference-only:
            if run_refine: returns (out_refine, out_1, fp_upd, bp_upd)
            else:         returns (out_1, fp_upd, bp_upd)
        """
        # Query side: initial similarity and SSP
        out_0 = self._similarity_logits(feat_q, fp, bp)
        ssfp_1, ssbp_1, asfp_1, asbp_1 = self._ssp(feat_q, out_0)
        # Update prototypes (query-driven)
        fp_1 = fp * 0.5 + ssfp_1 * 0.5
        bp_1 = ssbp_1 * 0.3 + asbp_1 * 0.7

        out_1 = self._similarity_logits(feat_q, fp_1, bp_1)

        if run_refine:
            # Another SSP/refine on query
            ssfp_2, ssbp_2, asfp_2, asbp_2 = self._ssp(feat_q, out_1)
            fp_2 = fp * 0.5 + ssfp_2 * 0.5
            bp_2 = ssbp_2 * 0.3 + asbp_2 * 0.7
            fp_2 = fp * 0.5 + fp_1 * 0.2 + fp_2 * 0.3
            bp_2 = bp * 0.5 + bp_1 * 0.2 + bp_2 * 0.3
            out_refine = self._similarity_logits(feat_q, fp_2, bp_2)
            out_refine = out_refine * 0.7 + out_1 * 0.3
            fp_upd, bp_upd = fp_2, bp_2
        else:
            fp_upd, bp_upd = fp_1, bp_1

        # (Optional) support-side SSP for alignment (inference-only, no loss)
        if feat_s is not None:
            # Repeat query prototypes for multi-shot alignment if needed
            if shot > 1:
                rep = shot
                fp_rep = fp_upd.repeat(rep, 1, 1, 1)  # [K, C, 1, 1]
                bp_rep = bp_upd.repeat(rep, 1, 1, 1)
            else:
                fp_rep, bp_rep = fp_upd, bp_upd

            supp_out_0 = self._similarity_logits(feat_s, fp_rep, bp_rep)
            ssfp_s, ssbp_s, asfp_s, asbp_s = self._ssp(feat_s, supp_out_0)
            fp_s = (fp_rep * 0.5 + ssfp_s * 0.5)
            bp_s = (ssbp_s * 0.3 + asbp_s * 0.7)

            # If multi-shot, average per-episode prototypes back to [B=K/shot]
            if shot > 1:
                # Average consecutive groups of size=shot across batch dimension
                k_total = fp_s.shape[0]
                assert k_total % shot == 0
                groups = k_total // shot
                fp_chunks = fp_s.view(groups, shot, *fp_s.shape[1:]).mean(dim=1)
                bp_chunks = bp_s.view(groups, shot, *bp_s.shape[1:]).mean(dim=1)
                fp_s, bp_s = fp_chunks, bp_chunks

            # Merge support-refined prototypes (keep simple averaging with query-updated)
            fp_upd = 0.5 * (fp_upd + fp_s.mean(dim=0, keepdim=True))
            bp_upd = 0.5 * (bp_upd + bp_s.mean(dim=0, keepdim=True))

        if run_refine:
            return out_refine, out_1, fp_upd, bp_upd
        else:
            return out_1, fp_upd, bp_upd

    def run_single_scale(
        self,
        feat_s_list: List[torch.Tensor],  # list of [1, C, Hs, Ws]
        mask_s_list: List[torch.Tensor],  # list of [Hs0, Ws0] (will be resized)
        feat_q: torch.Tensor,             # [1, C, Hq, Wq]
        iters: Optional[int] = None,
        use_refine: Optional[bool] = None,
        collect_history: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run IFA iteration on a single scale.
        Returns logits: [1, 2, Hq, Wq]. When `collect_history=True`, also returns
        the list of per-iteration logits (same spatial size).
        """
        iters = self.iters if iters is None else int(iters)
        run_refine = self.use_refine if use_refine is None else bool(use_refine)

        # Build initial support prototypes by masked average pooling, then average across K
        fg_list = []
        bg_list = []
        supp_feats_cat = []
        for f_s, m_s in zip(feat_s_list, mask_s_list):
            # Align mask to support feature size inside pooling fn
            fg = self._masked_average_pooling(f_s, (m_s == 1).float().unsqueeze(0))  # [1, C]
            bg = self._masked_average_pooling(f_s, (m_s == 0).float().unsqueeze(0))  # [1, C]
            fg_list.append(fg)
            bg_list.append(bg)
            supp_feats_cat.append(f_s)

        fp = torch.mean(torch.cat(fg_list, dim=0), dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # [1,C,1,1]
        bp = torch.mean(torch.cat(bg_list, dim=0), dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # [1,C,1,1]

        supp_feats = torch.cat(supp_feats_cat, dim=0) if len(supp_feats_cat) > 0 else None  # [K,C,H,W]

        # Iterative BFP
        out_main: Optional[torch.Tensor] = None
        fp_cur, bp_cur = fp, bp
        history: List[torch.Tensor] = [] if collect_history else []

        for t in range(iters):
            if t == 0 and run_refine:
                out_refine, out_1, fp_cur, bp_cur = self._iter_bfp(fp_cur, bp_cur, supp_feats, feat_q, True, shot=max(1, len(feat_s_list)))
                out_main = out_refine
            else:
                out_1, fp_cur, bp_cur = self._iter_bfp(fp_cur, bp_cur, supp_feats, feat_q, False, shot=max(1, len(feat_s_list)))
                out_main = out_1
            if collect_history:
                history.append(out_main.clone())

        assert out_main is not None
        if collect_history:
            return out_main, history
        return out_main

    def run_multi_scale(
        self,
        feats_s_ms: List[List[torch.Tensor]],  # per-scale list over K supports: [S][K][1,Cs,Hs,Ws]
        masks_s: List[torch.Tensor],           # K masks at image resolution [H,W]
        feats_q_ms: List[torch.Tensor],        # [S][1,Cs,Hs,Ws]
        out_size: Tuple[int, int],
        weights: Optional[List[float]] = None,
        iters: Optional[int] = None,
        use_refine: Optional[bool] = None,
        collect_history: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, List]]:
        """
        Multi-scale IFA: run per-scale, upsample logits to out_size, and fuse.
        Returns fused logits [1, 2, out_h, out_w].
        """
        s = len(feats_q_ms)
        if s == 0:
            raise ValueError("Empty feature list for multi-scale IFA.")
        if weights is None:
            weights = [1.0 / s] * s
        if len(weights) != s:
            raise ValueError("Length of weights must match number of scales.")

        fused = None
        per_scale_details: List[Dict[str, object]] = [] if collect_history else []
        fused_iter_logits: List[torch.Tensor] | None = [] if collect_history else None
        num_iters_expected: Optional[int] = None

        for si in range(s):
            # Build K support features at this scale
            feat_s_list = [feats_s_ms[si][k] for k in range(len(masks_s))]
            result = self.run_single_scale(
                feat_s_list,
                masks_s,
                feats_q_ms[si],
                iters=iters,
                use_refine=use_refine,
                collect_history=collect_history,
            )
            if collect_history:
                logits_si, history_si = result  # type: ignore[misc]
            else:
                logits_si = result  # type: ignore[assignment]
                history_si = None

            logits_si_up = F.interpolate(logits_si, size=out_size, mode="bilinear", align_corners=False)
            if fused is None:
                fused = weights[si] * logits_si_up
            else:
                fused = fused + weights[si] * logits_si_up

            if collect_history and history_si is not None:
                if num_iters_expected is None:
                    num_iters_expected = len(history_si)
                elif len(history_si) != num_iters_expected:
                    raise RuntimeError("Mismatch in iteration counts across scales.")

                per_scale_details.append(
                    {
                        "scale_index": si,
                        "spatial_size": list(feats_q_ms[si].shape[-2:]),
                        "final_logits": logits_si,
                        "iter_logits": history_si,
                    }
                )
                upsampled_hist = [
                    F.interpolate(h, size=out_size, mode="bilinear", align_corners=False) for h in history_si
                ]
                if fused_iter_logits is None or len(fused_iter_logits) == 0:
                    fused_iter_logits = [weights[si] * h for h in upsampled_hist]
                else:
                    if len(fused_iter_logits) != len(upsampled_hist):
                        raise RuntimeError("Fused-history accumulator size mismatch.")
                    fused_iter_logits = [
                        prev + weights[si] * h for prev, h in zip(fused_iter_logits, upsampled_hist)
                    ]

        assert fused is not None
        if collect_history:
            return fused, {
                "per_scale": per_scale_details,
                "fused_iter_logits": fused_iter_logits or [],
            }
        return fused
