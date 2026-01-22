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

import unittest

import paddle

# from fastdeploy.spec_decode.draft_tree_utils import tree_revert_cache


class TestTreeRevertCache(unittest.TestCase):
    def setUp(self) -> None:
        self.max_num_seqs = 2

        self.num_hidden_layers = 3
        self.num_blocks = 4
        self.kv_num_heads = 2
        self.block_size = 64
        self.head_dim = 128
        self.cache_type = paddle.bfloat16

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape[-1] = self.head_dim // 2
        value_cache_shape = key_cache_shape
        return key_cache_shape, value_cache_shape

    def init_kv_cache(self):
        key_cache_shape, value_cache_shape = self.get_kv_cache_shape(self.num_blocks)
        cache_kvs_list = []
        for i in range(self.num_hidden_layers):
            key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=self.cache_type)
            val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=self.cache_type)
            cache_kvs_list.extend([key_cache, val_cache])

        return cache_kvs_list

    def test_tree_revert_cache(self):
        caches = self.init_kv_cache()
        print(caches)


if __name__ == "__main__":
    unittest.main()
