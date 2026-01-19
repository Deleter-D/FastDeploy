// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"
#include "paddle/extension.h"

__global__ void build_tree_kernel(const int64_t* top_scores_idx,
                                  const int64_t* parent_list,
                                  const int* seq_lens_verified,
                                  bool* tree_mask,
                                  int64_t* positions,
                                  int64_t* retrive_index,
                                  int64_t* retrive_next_token,
                                  int64_t* retrive_next_sibling,
                                  const int tree_topk,
                                  const int depth,
                                  const int draft_token_num,
                                  const int max_model_len) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid > draft_token_num) return;
  int seq_len = seq_lens_verified[bid];

  int tree_mask_len = max_model_len + draft_token_num + 1;
  int tree_mask_start_idx = bid * tree_mask_len * tree_mask_len +
                            max_model_len * tree_mask_len + max_model_len + 1;

  // visible for self, mask its siblings

  // printf(
  //     "bid: %d, tid: %d, seq_len: %d, "
  //     "tree_mask_start_idx: %d\n",
  //     bid,
  //     tid,
  //     seq_len,
  //     tree_mask_start_idx);
  int position = 0;
  if (tid == 0) {
    // tree_mask[tree_mask_start_idx - 1] = false;
    for (int i = 0; i < draft_token_num + 1; i++) {
      tree_mask[(tree_mask_start_idx - 1) + i * tree_mask_len] = false;
    }

    positions[bid * (draft_token_num + 1)] = seq_len;
    // int retrive_index_offset = bid * (draft_token_num + 1);
    int retrive_index_offset = 0;
    for (int i = draft_token_num; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * (draft_token_num + 1) + i] = current_token_idx;
      // get current token's parent tree base index
      int64_t parent_tb_idx =
          top_scores_idx[bid * draft_token_num + i - 1] / tree_topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int64_t parent_token_idx =
            parent_list[bid * (tree_topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < (draft_token_num + 1); ++parent_position) {
          if (top_scores_idx[bid * draft_token_num + parent_position] ==
              parent_token_idx) {
            // current token's parent is selected,
            // this is a root-traceable path
            ++parent_position;
            break;
          }
        }
      }
      // printf(
      //     "tid: 0, draft_token: %d, "
      //     "parent_tb_idx: %lld, parent_position: %d\n",
      //     i,
      //     parent_tb_idx,
      //     parent_position);
      if (retrive_next_token[bid * (draft_token_num + 1) + parent_position] ==
          -1) {
        // current token's parent has not successor
        retrive_next_token[bid * (draft_token_num + 1) + parent_position] = i;
      } else {
        // current token's parent already has successor
        int64_t origin_next_token =
            retrive_next_token[bid * (draft_token_num + 1) + parent_position];
        retrive_next_token[bid * (draft_token_num + 1) + parent_position] = i;
        retrive_next_sibling[bid * (draft_token_num + 1) + i] =
            origin_next_token;
      }
    }
    // retrive_index[bid * (draft_token_num + 1)] =
    //     bid * (draft_token_num + 1);
    retrive_index[bid * (draft_token_num + 1)] = 0;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      // visible for its ancestors
      tree_mask[tree_mask_start_idx + tid * tree_mask_len + cur_position] =
          false;
      int64_t parent_tb_idx =
          top_scores_idx[bid * draft_token_num + cur_position] / tree_topk;
      printf(
          "bid: %d, tid: %d, cur_position: %d, top_scores_idx: %lld, "
          "tree_topk: %d, "
          "parent_tb_idx: %lld\n",
          bid,
          tid,
          cur_position,
          top_scores_idx[bid * draft_token_num + cur_position],
          tree_topk,
          parent_tb_idx);
      if (parent_tb_idx == 0) {
        break;
      }
      int64_t token_idx =
          parent_list[bid * (tree_topk * (depth - 1) + 1) + parent_tb_idx];
      printf("bid: %d, tid: %d, token_idx: %lld\n", bid, tid, token_idx);
      for (cur_position = 0; cur_position < (draft_token_num + 1);
           ++cur_position) {
        if (top_scores_idx[bid * draft_token_num + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * (draft_token_num + 1) + tid] = position + seq_len;
  }
}

void BuildTree(const paddle::Tensor& top_scores_idx,
               const paddle::Tensor& parent_list,
               const paddle::Tensor& seq_lens_verified,
               const paddle::Tensor& tree_mask,
               const paddle::Tensor& positions,
               const paddle::Tensor& retrive_index,
               const paddle::Tensor& retrive_next_token,
               const paddle::Tensor& retrive_next_sibling,
               const int tree_topk,
               const int depth,
               const int draft_token_num,
               const int max_model_len) {
  int bsz = parent_list.shape()[0];
  auto cur_stream = seq_lens_verified.stream();
  // printf("bsz: %d, tree_topk: %d, draft_token_num: %d\n",
  //        bsz,
  //        tree_topk,
  //        draft_token_num);
  dim3 grid(bsz);
  dim3 block(draft_token_num + 1);
  build_tree_kernel<<<grid, block, 0, cur_stream>>>(
      top_scores_idx.data<int64_t>(),
      parent_list.data<int64_t>(),
      seq_lens_verified.data<int>(),
      const_cast<bool*>(tree_mask.data<bool>()),
      const_cast<int64_t*>(positions.data<int64_t>()),
      const_cast<int64_t*>(retrive_index.data<int64_t>()),
      const_cast<int64_t*>(retrive_next_token.data<int64_t>()),
      const_cast<int64_t*>(retrive_next_sibling.data<int64_t>()),
      tree_topk,
      depth,
      draft_token_num,
      max_model_len);
}

PD_BUILD_STATIC_OP(eagle_build_tree)
    .Inputs({"top_scores_idx",
             "parent_list",
             "seq_lens_verified",
             "tree_mask",
             "positions",
             "retrive_index",
             "retrive_next_token",
             "retrive_next_sibling"})
    .Attrs({"tree_topk: int",
            "depth: int",
            "draft_token_num: int",
            "max_model_len: int"})
    .Outputs({"tree_mask_out",
              "positions_out",
              "retrive_index_out",
              "retrive_next_token_out",
              "retrive_next_sibling_out"})
    .SetInplaceMap({{"tree_mask", "tree_mask_out"},
                    {"positions", "positions_out"},
                    {"retrive_index", "retrive_index_out"},
                    {"retrive_next_token", "retrive_next_token_out"},
                    {"retrive_next_sibling", "retrive_next_sibling_out"}})
    .SetKernelFn(PD_KERNEL(BuildTree));

__global__ void verify_tree_greedy_kernel(int64_t* accept_tokens,
                                          int* accept_num,
                                          int64_t* step_idx,
                                          bool* stop_flags,
                                          const int64_t* draft_tokens,
                                          const int64_t* retrive_index,
                                          const int64_t* retrive_next_token,
                                          const int64_t* retrive_next_sibling,
                                          int64_t* accepted_retrive_index,
                                          const int64_t* verify_tokens,
                                          const int* seq_lens_encoder,
                                          const int* seq_lens_decoder,
                                          const int* seq_lens_this_time,
                                          int* seq_lens_verified,
                                          const int* actual_candidate_len,
                                          const int64_t* max_dec_len,
                                          const int64_t* end_tokens,
                                          const bool* is_block_step,
                                          const int* output_cum_offsets,
                                          const int real_bsz,
                                          const int max_draft_tokens,
                                          const int speculate_steps,
                                          const int end_length,
                                          const int max_seq_len,
                                          const int max_candidate_len,
                                          const bool prefill_one_step_stop) {
  int bid = threadIdx.x;
  int accept_num_now = 1;
  int stop_flag_now_int = 0;

  if (!(is_block_step[bid] || bid >= real_bsz)) {
    const int start_token_id = bid * max_seq_len - output_cum_offsets[bid];
    printf("[verify][index] start_token_id: %d\n", start_token_id);

    if (stop_flags[bid]) {
      stop_flag_now_int = 1;
    } else {
      auto* verify_tokens_now =
          verify_tokens + start_token_id * max_candidate_len;
      auto* draft_tokens_now = draft_tokens + bid * max_draft_tokens;
      auto* actual_candidate_len_now = actual_candidate_len + start_token_id;

      auto* retrive_index_now = retrive_index + bid * max_draft_tokens;
      int64_t last_accepted_retrive_idx = retrive_index_now[0];

      int cur_index = 0;
      int i = 0;
      int64_t draft_idx = 0;
      int64_t draft_token;
      int64_t verify_token = verify_tokens_now[i * max_candidate_len];
      bool no_accept_all_layer = true;
      for (; i < speculate_steps; ++i) {
        if (seq_lens_encoder[bid] != 0) {
          seq_lens_verified[bid] = seq_lens_encoder[bid];
          break;
        }
        bool no_accept_this_layer = true;
        cur_index = retrive_next_token[bid * max_draft_tokens + cur_index];
        while (cur_index != -1) {
          draft_idx = retrive_index_now[cur_index];
          draft_token = draft_tokens_now[cur_index];
          verify_token = verify_tokens_now[i * max_candidate_len];

          printf(
              "[verify] bsz: %d, cur_index: %d, "
              "draft_idx: %lld, draft_token: %lld, "
              "verify_token: %lld\n",
              bid,
              cur_index,
              draft_idx,
              draft_token,
              verify_token);
          if (draft_token == verify_token) {
            step_idx[bid]++;
            accept_tokens[bid * speculate_steps + i] = draft_token;
            no_accept_this_layer = false;
            last_accepted_retrive_idx = draft_idx;
            accepted_retrive_index[bid * max_draft_tokens + i] = draft_idx;
            printf(
                "[verify][verify stage] bsz: %d, "
                "accept_token: %lld\n",
                bid,
                draft_token);
            if (is_in_end(draft_token, end_tokens, end_length) ||
                step_idx[bid] >= max_dec_len[bid]) {
              stop_flags[bid] = true;
              stop_flag_now_int = 1;
              if (step_idx[bid] >= max_dec_len[bid]) {
                accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
              }
            } else {
              accept_num_now++;
            }
            break;
          } else {
            cur_index =
                retrive_next_sibling[bid * max_draft_tokens + cur_index];
          }
        }

        if (no_accept_this_layer || stop_flag_now_int || stop_flags[bid]) {
          break;
        }
        no_accept_all_layer = no_accept_all_layer && no_accept_this_layer;
      }

      if (!stop_flag_now_int) {
        step_idx[bid]++;
        int64_t accept_token;
        if (no_accept_all_layer) {
          accept_token = verify_token;
          printf(
              "[verify][sampling stage] bsz: %d, "
              "none token accepted, "
              "accept_token: %lld\n",
              bid,
              accept_token);
        } else {
          accept_token = verify_tokens_now[draft_idx * max_candidate_len];
          printf(
              "[verify][sampling stage] bsz: %d, "
              "has token accepted, "
              "i: %d, draft_idx: %lld, accept_token: %lld\n",
              bid,
              i,
              draft_idx,
              accept_token);
        }
        accept_tokens[bid * max_draft_tokens + i] = accept_token;

        if (prefill_one_step_stop) {
          stop_flags[bid] = true;
        }

        if (is_in_end(accept_token, end_tokens, end_length) ||
            step_idx[bid] >= max_dec_len[bid]) {
          stop_flags[bid] = true;
          stop_flag_now_int = 1;
          if (step_idx[bid] >= max_dec_len[bid]) {
            accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
          }
        }
      }

      accept_num[bid] = accept_num_now;
      if (seq_lens_encoder[bid] == 0) {
        seq_lens_verified[bid] += accept_num_now;
      }
    }
  }
}

void SpeculateVerifyTreeGreedy(const paddle::Tensor& accept_tokens,
                               const paddle::Tensor& accept_num,
                               const paddle::Tensor& step_idx,
                               const paddle::Tensor& stop_flags,
                               const paddle::Tensor& draft_tokens,
                               const paddle::Tensor& retrive_index,
                               const paddle::Tensor& retrive_next_token,
                               const paddle::Tensor& retrive_next_sibling,
                               const paddle::Tensor& accepted_retrive_index,
                               const paddle::Tensor& verify_tokens,
                               const paddle::Tensor& seq_lens_encoder,
                               const paddle::Tensor& seq_lens_decoder,
                               const paddle::Tensor& seq_lens_this_time,
                               const paddle::Tensor& seq_lens_verified,
                               const paddle::Tensor& actual_candidate_len,
                               const paddle::Tensor& max_dec_len,
                               const paddle::Tensor& end_tokens,
                               const paddle::Tensor& is_block_step,
                               const paddle::Tensor& output_cum_offsets,
                               const int speculate_steps,
                               const int max_seq_len) {
  auto bsz = accept_tokens.shape()[0];
  auto real_bsz = seq_lens_this_time.shape()[0];
  auto max_draft_tokens = draft_tokens.shape()[1];
  auto end_length = end_tokens.shape()[0];
  auto max_candidate_len = verify_tokens.shape()[1];

  bool prefill_one_step_stop = false;
  if (const char* env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP")) {
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }

  printf(
      "bsz: %d, real_bsz: %d, max_draft_tokens: %d, end_length: %d, "
      "max_candidate_len: %d, speculate_steps: %d\n",
      bsz,
      real_bsz,
      max_draft_tokens,
      end_length,
      max_candidate_len,
      speculate_steps);

  verify_tree_greedy_kernel<<<1, 512, 0, accept_tokens.stream()>>>(
      const_cast<int64_t*>(accept_tokens.data<int64_t>()),
      const_cast<int*>(accept_num.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      const_cast<bool*>(stop_flags.data<bool>()),
      draft_tokens.data<int64_t>(),
      retrive_index.data<int64_t>(),
      retrive_next_token.data<int64_t>(),
      retrive_next_sibling.data<int64_t>(),
      const_cast<int64_t*>(accepted_retrive_index.data<int64_t>()),
      verify_tokens.data<int64_t>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      seq_lens_this_time.data<int>(),
      const_cast<int*>(seq_lens_verified.data<int>()),
      actual_candidate_len.data<int>(),
      max_dec_len.data<int64_t>(),
      end_tokens.data<int64_t>(),
      is_block_step.data<bool>(),
      output_cum_offsets.data<int>(),
      real_bsz,
      max_draft_tokens,
      speculate_steps,
      end_length,
      max_seq_len,
      max_candidate_len,
      prefill_one_step_stop);
}

PD_BUILD_STATIC_OP(speculate_verify_tree_greedy)
    .Inputs({"accept_tokens",
             "accept_num",
             "step_idx",
             "stop_flags",
             "draft_tokens",
             "retrive_index",
             "retrive_next_token",
             "retrive_next_sibling",
             "accepted_retrive_index",
             "verify_tokens",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "seq_lens_verified",
             "actual_candidate_len",
             "max_dec_len",
             "end_tokens",
             "is_block_step",
             "output_cum_offsets"})
    .Attrs({"speculate_steps: int", "max_seq_len: int"})
    .Outputs({"accept_tokens_out",
              "accept_num_out",
              "step_idx_out",
              "stop_flags_out",
              "seq_lens_verified_out",
              "accepted_retrive_index_out"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"},
                    {"step_idx", "step_idx_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_verified", "seq_lens_verified_out"},
                    {"accepted_retrive_index", "accepted_retrive_index_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateVerifyTreeGreedy));
