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

template <typename T>
__global__ void tree_revert_cache_kernel(
    T* cache_k,  // [max_block_num, num_heads, block_size,
                 // head_dim]
    T* cache_v,
    const int* seq_lens_this_time,
    const int* seq_lens_decoder,
    const int* seq_lens_encoder,
    const int* block_table,  // [bsz, block_num_per_seq]
) {}

void TreeRevertCache(const paddle::Tensor& cache_k,
                     const paddle::Tensor& cache_v,
                     const paddle::Tensor& seq_lens_this_time,
                     const paddle::Tensor& seq_lens_decoder,
                     const paddle::Tensor& seq_lens_encoder,
                     const paddle::Tensor& block_table) {}

PD_BUILD_STATIC_OP(tree_revert_cache)

PD_BUILD_STATIC_OP(tree_revert_cache)
    .Inputs({})
    .Attrs({})
    .Outputs({})
    .SetInplaceMap({
        {},
    })
    .SetKernelFn(PD_KERNEL(TreeRevertCache));
