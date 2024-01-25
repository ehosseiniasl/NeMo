# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin


class GreedyBatchedRNNTLoopLabelsComputer(nn.Module, ConfidenceMethodMixin):
    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)

    def forward(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ):
        """
        Optimized batched greedy decoding.
        The main idea: search for next labels for the whole batch (evaluating Joint)
        and thus always evaluate prediction network with maximum possible batch size
        """
        batch_size, max_time, _unused = x.shape
        device = x.device

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # Initialize empty hypotheses and all necessary tensors
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size, init_length=max_time, device=x.device, float_dtype=x.dtype
        )
        time_indices = torch.zeros([batch_size], dtype=torch.long, device=device)  # always of batch_size
        active_indices = torch.arange(batch_size, dtype=torch.long, device=device)  # initial: all indices
        labels = torch.full([batch_size], fill_value=self._blank_index, dtype=torch.long, device=device)

        # init additional structs for hypotheses: last decoder state, alignments, frame_confidence

        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(x)

        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2,  # blank for each timestep + text tokens
            device=x.device,
            float_dtype=x.dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
        )

        # loop while there are active indices
        first_step = True
        state = self.decoder.initialize_state(torch.zeros(batch_size, device=device, dtype=x.dtype))
        while active_indices.shape[0] > 0:
            current_batch_size = active_indices.shape[0]
            # stage 1: get decoder (prediction network) output
            if first_step:
                # start of the loop, SOS symbol is passed into prediction network, state is None
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), None, add_sos=False, batch_size=current_batch_size
                )
                first_step = False
            else:
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), state, add_sos=False, batch_size=current_batch_size
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self.joint.joint_after_projection(
                    x[active_indices, time_indices[active_indices]].unsqueeze(1), decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            if self.preserve_frame_confidence:
                logits = F.log_softmax(logits, dim=-1)
            scores, labels = logits.max(-1)

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            if use_alignments:
                alignments.add_results_(
                    active_indices=active_indices,
                    time_indices=time_indices[active_indices],
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                )
            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            advance_mask = torch.logical_and(blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices]))
            while advance_mask.any():
                advance_indices = active_indices[advance_mask]
                time_indices[advance_indices] += 1
                logits = (
                    self.joint.joint_after_projection(
                        x[advance_indices, time_indices[advance_indices]].unsqueeze(1), decoder_output[advance_mask],
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                if self.preserve_frame_confidence:
                    logits = F.log_softmax(logits, dim=-1)

                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits.max(-1)
                labels[advance_mask] = more_labels
                scores[advance_mask] = more_scores
                if use_alignments:
                    alignments.add_results_(
                        active_indices=advance_indices,
                        time_indices=time_indices[advance_indices],
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                    )
                blank_mask = labels == self._blank_index
                advance_mask = torch.logical_and(
                    blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices])
                )

            # stage 3: filter labels and state, store hypotheses
            # the only case, when there are blank labels in predictions - when we found the end for some utterances
            if blank_mask.any():
                non_blank_mask = ~blank_mask
                labels = labels[non_blank_mask]
                scores = scores[non_blank_mask]

                # select states for hyps that became inactive (is it necessary?)
                # this seems to be redundant, but used in the `loop_frames` output
                inactive_global_indices = active_indices[blank_mask]
                inactive_inner_indices = torch.arange(current_batch_size, device=device, dtype=torch.long)[blank_mask]
                self.decoder.batch_replace_states(
                    src_states=state,
                    src_mask_or_indices=inactive_inner_indices,
                    dst_states=last_decoder_state,
                    dst_mask_or_indices=inactive_global_indices,
                )

                # update active indices and state
                active_indices = active_indices[non_blank_mask]
                state = self.decoder.mask_select_states(state, non_blank_mask)
            # store hypotheses
            batched_hyps.add_results_(
                active_indices, labels, time_indices[active_indices].clone(), scores,
            )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    torch.logical_and(
                        labels != self._blank_index,
                        batched_hyps.last_timestep_lasts[active_indices] >= self.max_symbols,
                    ),
                    batched_hyps.last_timestep[active_indices] == time_indices[active_indices],
                )
                if force_blank_mask.any():
                    # forced blank is not stored in the alignments following the original implementation
                    time_indices[active_indices[force_blank_mask]] += 1  # emit blank => advance time indices
                    # elements with time indices >= out_len become inactive, remove them from batch
                    still_active_mask = time_indices[active_indices] < out_len[active_indices]
                    active_indices = active_indices[still_active_mask]
                    labels = labels[still_active_mask]
                    state = self.decoder.mask_select_states(state, still_active_mask)

        if use_alignments:
            return batched_hyps, alignments, last_decoder_state
        return batched_hyps, None, last_decoder_state

    @torch.jit.export
    def forward_const_batch_size(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ):
        """
        Optimized batched greedy decoding.
        The main idea: search for next labels for the whole batch (evaluating Joint)
        and thus always evaluate prediction network with maximum possible batch size
        """
        batch_size, max_time, _unused = x.shape
        device = x.device

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # Initialize empty hypotheses and all necessary tensors
        assert self.max_symbols is not None
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size, init_length=max_time * self.max_symbols, device=x.device, float_dtype=x.dtype
        )
        time_indices = torch.zeros([batch_size], dtype=torch.long, device=device)
        safe_time_indices = torch.zeros_like(time_indices)
        all_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = out_len - 1
        labels = torch.full([batch_size], fill_value=self._blank_index, dtype=torch.long, device=device)

        # init additional structs for hypotheses: last decoder state, alignments, frame_confidence

        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(x)

        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2,  # blank for each timestep + text tokens
            device=x.device,
            float_dtype=x.dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
        )

        # loop while there are active indices
        first_step = True
        state = self.decoder.initialize_state(torch.zeros(batch_size, device=device, dtype=x.dtype))
        active_mask: torch.Tensor = out_len > 0
        active_mask_prev = torch.empty_like(active_mask)
        became_inactive_mask = torch.empty_like(active_mask)
        advance_mask = torch.empty_like(active_mask)

        while active_mask.any():
            # torch.cuda.set_sync_debug_mode(2)
            active_mask_prev.copy_(active_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            if first_step:
                # start of the loop, SOS symbol is passed into prediction network, state is None
                # we need to separate this for torch.jit
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), None, add_sos=False, batch_size=batch_size
                )
                first_step = False
            else:
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self.joint.joint_after_projection(x[all_indices, safe_time_indices].unsqueeze(1), decoder_output,)
                .squeeze(1)
                .squeeze(1)
            )
            if self.preserve_frame_confidence:
                logits = F.log_softmax(logits, dim=-1)
            scores, labels = logits.max(-1)

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            time_indices_current_labels.copy_(time_indices, non_blocking=True)
            # if use_alignments:
            #     alignments.add_results_(
            #         active_indices=active_indices,
            #         time_indices=time_indices[active_indices],
            #         logits=logits if self.preserve_alignments else None,
            #         labels=labels if self.preserve_alignments else None,
            #         confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
            #     )
            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            time_indices += blank_mask
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, out_len, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)
            # torch.cuda.set_sync_debug_mode(0)
            while advance_mask.any():
                # torch.cuda.set_sync_debug_mode(2)
                logits = (
                    self.joint.joint_after_projection(x[all_indices, safe_time_indices].unsqueeze(1), decoder_output,)
                    .squeeze(1)
                    .squeeze(1)
                )
                if self.preserve_frame_confidence:
                    logits = F.log_softmax(logits, dim=-1)

                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits.max(-1)
                # labels[advance_mask] = more_labels
                torch.where(advance_mask, more_labels, labels, out=labels)
                # scores[advance_mask] = more_scores
                torch.where(advance_mask, more_scores, scores, out=scores)
                # if use_alignments:
                #     alignments.add_results_(
                #         active_indices=advance_indices,
                #         time_indices=time_indices[advance_indices],
                #         logits=logits if self.preserve_alignments else None,
                #         labels=more_labels if self.preserve_alignments else None,
                #         confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                #     )
                blank_mask = labels == self._blank_index
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                time_indices += blank_mask
                safe_time_indices = torch.minimum(time_indices, last_timesteps)
                torch.less(time_indices, out_len, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)
                # torch.cuda.set_sync_debug_mode(0)
            # torch.cuda.set_sync_debug_mode(2)

            # stage 3: filter labels and state, store hypotheses
            # select states for hyps that became inactive (is it necessary?)
            # this seems to be redundant, but used in the `loop_frames` output
            # torch.cuda.set_sync_debug_mode(2)
            torch.ne(active_mask, active_mask_prev, out=became_inactive_mask)
            self.decoder.batch_replace_states_mask(
                src_states=state, dst_states=last_decoder_state, mask=became_inactive_mask,
            )
            # torch.cuda.set_sync_debug_mode(0)

            # store hypotheses
            batched_hyps.add_results_masked_no_checks_(
                active_mask, labels, time_indices_current_labels, scores,
            )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index, batched_hyps.last_timestep_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestep == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, out_len, out=active_mask)
            # torch.cuda.set_sync_debug_mode(0)
        if use_alignments:
            return batched_hyps, alignments, last_decoder_state
        return batched_hyps, None, last_decoder_state
