"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

from typing import List

import paddle

from fastdeploy.model_executor.ops.gpu import eagle_build_tree


def tree_select_top_k(
    depth: int,
    topk_probs: paddle.Tensor,
    topk_token_ids: paddle.Tensor,
    path_scores: paddle.Tensor,
    hidden_states: paddle.Tensor,
    tree_topk: int,
):
    # print(f"[TreeSelectTopK][Input] topk_probs: {topk_probs}")
    # print(f"[TreeSelectTopK][Input] topk_token_ids: {topk_token_ids}")
    # print(f"[TreeSelectTopK][Input] path_scores: {path_scores}")
    if depth == 0:
        topk_token_ids_res = topk_token_ids
        path_scores = topk_probs
        tree_info = (
            topk_probs.unsqueeze(1),
            topk_token_ids,
            paddle.arange(-1, tree_topk, dtype="int64").unsqueeze(0).repeat_interleave(topk_probs.shape[0], 0),
        )
        target_hidden_states = hidden_states.repeat_interleave(repeats=tree_topk, axis=0)
    else:
        scores = paddle.multiply(path_scores.unsqueeze(2), topk_probs.reshape((-1, tree_topk, tree_topk)))
        # print(f"[TreeSelectTopK][Score] scores: {scores}")
        topk_probs, topk_idx = paddle.topk(scores.flatten(start_axis=1), tree_topk, axis=-1)
        path_scores = topk_probs
        topk_token_ids = topk_token_ids.reshape((-1, tree_topk**2))
        # print(f"[TreeSelectTopK][Gather] topk_token_ids: {topk_token_ids}")
        # print(f"[TreeSelectTopK][Gather] topk_idx: {topk_idx}")
        topk_token_ids_res = paddle.take_along_axis(topk_token_ids, topk_idx, axis=1)
        # print(f"[TreeSelectTopK][Gather] topk_token_ids_res: {topk_token_ids_res}")

        tree_info = (
            scores,
            topk_token_ids,
            topk_idx + (tree_topk**2 * (depth - 1) + tree_topk),
        )

        selected_idx = topk_idx // tree_topk + paddle.arange(0, hidden_states.shape[0], step=tree_topk).unsqueeze(1)
        # print(f"[TreeSelectTopK][HiddenStates] selected_idx: {selected_idx}")
        target_hidden_states = hidden_states[selected_idx, :].reshape((-1, hidden_states.shape[-1]))
    return topk_token_ids_res, path_scores, tree_info, target_hidden_states


def build_tree_preprocess(
    verified_token_ids: paddle.Tensor,
    score_list: List[paddle.Tensor],
    token_list: List[paddle.Tensor],
    parents_list: List[paddle.Tensor],
    real_bsz: int,
    max_draft_token_num: int,
):
    score_list = paddle.concat(score_list, axis=1).flatten(1)
    print(f"[MTPProposer][TreePreprocess] score_list: {score_list}")
    token_list = paddle.concat(token_list, axis=1)
    print(f"[MTPProposer][TreePreprocess] token_list: {token_list}")

    _, top_scores_idx = paddle.topk(score_list, max_draft_token_num, axis=-1)
    top_scores_idx = paddle.sort(top_scores_idx)
    print(f"[MTPProposer][TreePreprocess] top_scores_idx: {top_scores_idx}")

    # draft_tokens = None
    # for token_ids, idx in zip(token_list, top_scores_idx):
    #     tmp = paddle.gather(token_ids, index=idx, axis=0)
    #     if draft_tokens is None:
    #         draft_tokens = tmp
    #         draft_tokens = paddle.unsqueeze(draft_tokens, axis=0)
    #     else:
    #         draft_tokens = paddle.stack([draft_tokens, tmp], axis=0)
    expanded_idx = paddle.unsqueeze(top_scores_idx, axis=-1)
    print(f"[MTPProposer][TreePreprocess] token_list: {token_list}")
    print(f"[MTPProposer][TreePreprocess] expanded_idx: {expanded_idx}")
    draft_tokens = paddle.take_along_axis(token_list.unsqueeze(axis=1), expanded_idx, axis=-1).squeeze(axis=-1)
    print(f"[MTPProposer][TreePreprocess] draft_tokens: {draft_tokens}")
    print(f"[MTPProposer][TreePreprocess] verified_token_ids: {verified_token_ids}")
    draft_tokens = paddle.concat([verified_token_ids[:real_bsz, :], draft_tokens], axis=1)
    print(f"[MTPProposer][TreePreprocess] draft_tokens: {draft_tokens}")

    if len(parents_list) > 1:
        parents_list = paddle.concat(parents_list[:-1], axis=1)
    else:
        parents_list = paddle.empty([parents_list[0].shape[0], 0], dtype="int64")

    return parents_list, top_scores_idx, draft_tokens


def build_tree(
    score_list: List[paddle.Tensor],
    token_list: List[paddle.Tensor],
    parents_list: List[paddle.Tensor],
    seq_lens_this_time: paddle.Tensor,
    base_model_draft_tokens: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    retrive_index: paddle.Tensor,
    retrive_next_token: paddle.Tensor,
    retrive_next_sibling: paddle.Tensor,
    seq_lens_verified: paddle.Tensor,
    tree_mask: paddle.Tensor,
    max_draft_token_num: int,
    num_model_steps: int,
    tree_topk: int,
    max_model_len: int,
):
    real_bsz = seq_lens_this_time.shape[0]
    print(f"[MTPProposer][BuildTree] real_bsz: {real_bsz}")

    verified_token_ids = base_model_draft_tokens[:, :1]
    parents_list, top_scores_idx, draft_tokens_res = build_tree_preprocess(
        verified_token_ids, score_list, token_list, parents_list, real_bsz, max_draft_token_num
    )
    print(f"[MTPProposer][BuildTree] parents_list: {parents_list}")
    print(f"[MTPProposer][BuildTree] top_scores_idx: {top_scores_idx}")
    print(f"[MTPProposer][BuildTree] draft_tokens: {draft_tokens_res}")

    positions = paddle.empty((real_bsz * (max_draft_token_num + 1),), dtype="int64")
    retrive_index[:] = -1
    retrive_next_token[:] = -1
    retrive_next_sibling[:] = -1
    eagle_build_tree(
        top_scores_idx,
        parents_list,
        seq_lens_verified,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        tree_topk,
        num_model_steps,
        max_draft_token_num,
        max_model_len,
    )

    draft_tokens.copy_(draft_tokens_res, True)
    print(f"[MTPProposer][BuildTree] positions: {positions}")
    print(f"[MTPProposer][BuildTree] retrive_index: {retrive_index}")
    print(f"[MTPProposer][BuildTree] retrive_next_token: {retrive_next_token}")
    print(f"[MTPProposer][BuildTree] retrive_next_sibling: {retrive_next_sibling}")
    print(f"[MTPProposer][BuildTree] tree_mask: {tree_mask}")
