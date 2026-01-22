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
  int tree_mask_batch_offset = bid * tree_mask_len * tree_mask_len;
  int tree_mask_row_start_idx =
      tree_mask_batch_offset + seq_len * tree_mask_len;
  int tree_mask_start_idx = tree_mask_row_start_idx + seq_len + 1;

  // visible for verified tokens
  for (int i = 0; i < seq_len; i++) {
    tree_mask[tree_mask_row_start_idx + tid * tree_mask_len + i] = false;
  }

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
