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

import numpy as np
import paddle

from fastdeploy.spec_decode.draft_tree_utils import build_tree


class TestBuildTree(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_build_tree(self):
        # Input
        batch_size = 2
        max_model_len = 4
        max_draft_token_num = 3
        num_model_steps = 2
        tree_topk = 4
        score_list = [
            paddle.to_tensor([[[0.9, 0.8, 0.7, 0.6]], [[0.5, 0.4, 0.3, 0.2]]], dtype="float32"),
            paddle.to_tensor(
                [
                    [
                        [0.81, 0.09, 0.09, 0.09],
                        [0.64, 0.08, 0.08, 0.08],
                        [0.49, 0.07, 0.07, 0.07],
                        [0.36, 0.06, 0.06, 0.06],
                    ],
                    [
                        [0.25, 0.05, 0.05, 0.05],
                        [0.16, 0.04, 0.04, 0.04],
                        [0.09, 0.03, 0.03, 0.03],
                        [0.04, 0.02, 0.02, 0.02],
                    ],
                ],
                dtype="float32",
            ),
        ]
        token_list = [
            paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64"),
            paddle.to_tensor(
                [
                    [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44],
                    [51, 52, 53, 54, 61, 62, 63, 64, 71, 72, 73, 74, 81, 82, 83, 84],
                ],
                dtype="int64",
            ),
        ]
        parents_list = [
            paddle.to_tensor([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype="int64"),
            paddle.to_tensor([[4, 8, 12, 16], [4, 8, 12, 16]], dtype="int64"),
        ]

        seq_lens_this_time = paddle.to_tensor([[4], [4]], dtype="int32")
        base_model_draft_tokens = paddle.to_tensor([[1042, 1042, 1042, 1042], [1042, 1042, 1042, 1042]], dtype="int64")
        seq_lens_verified = paddle.to_tensor([[3], [3]], dtype="int32")

        # Output
        draft_tokens = paddle.to_tensor([[-1, -1, -1, -1], [-1, -1, -1, -1]], dtype="int64")
        retrive_index = paddle.full(shape=[batch_size, max_draft_token_num + 1], fill_value=-1, dtype="int64")
        retrive_next_token = paddle.full(shape=[batch_size, max_draft_token_num + 1], fill_value=-1, dtype="int64")
        retrive_next_sibling = paddle.full(shape=[batch_size, max_draft_token_num + 1], fill_value=-1, dtype="int64")
        tree_mask = paddle.full(
            shape=[
                batch_size,
                1,
                max_model_len + max_draft_token_num + 1,
                max_model_len + max_draft_token_num + 1,
            ],
            fill_value=True,
            dtype="bool",
        )

        for idx in range(seq_lens_this_time.shape[0]):
            tree_mask[idx, 0, : seq_lens_verified[idx, 0], : seq_lens_verified[idx, 0]] = paddle.logical_not(
                paddle.tril(paddle.ones(shape=(seq_lens_verified[idx, 0], seq_lens_verified[idx, 0])).astype("bool"))
            )

        build_tree(
            score_list,
            token_list,
            parents_list,
            seq_lens_this_time,
            base_model_draft_tokens,
            draft_tokens,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            seq_lens_verified,
            tree_mask,
            max_draft_token_num,
            num_model_steps,
            tree_topk,
            max_model_len,
        )

        paddle.set_printoptions(threshold=100000000)
        print(tree_mask[:, :, :7, :7])

        ref_retrive_index = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype="int64")
        ref_retrive_next_token = np.array([[1, 3, -1, -1], [1, -1, -1, -1]], dtype="int64")
        ref_retrive_next_sibling = np.array([[-1, 2, -1, -1], [-1, 2, 3, -1]], dtype="int64")
        ref_tree_mask = np.array(
            [
                [
                    [
                        [False, True, True, True, True, True, True, True],
                        [False, False, True, True, True, True, True, True],
                        [False, False, False, True, True, True, True, True],
                        [False, False, False, False, True, True, True, True],
                        [False, False, False, False, False, True, True, True],
                        [False, False, False, False, True, False, True, True],
                        [False, False, False, False, False, True, False, True],
                        [True, True, True, True, True, True, True, True],
                    ]
                ],
                [
                    [
                        [False, True, True, True, True, True, True, True],
                        [False, False, True, True, True, True, True, True],
                        [False, False, False, True, True, True, True, True],
                        [False, False, False, False, True, True, True, True],
                        [False, False, False, False, False, True, True, True],
                        [False, False, False, False, True, False, True, True],
                        [False, False, False, False, True, True, False, True],
                        [True, True, True, True, True, True, True, True],
                    ]
                ],
            ],
            dtype="bool",
        )

        np.testing.assert_array_equal(retrive_index.numpy(), ref_retrive_index)
        np.testing.assert_array_equal(retrive_next_token.numpy(), ref_retrive_next_token)
        np.testing.assert_array_equal(retrive_next_sibling.numpy(), ref_retrive_next_sibling)
        np.testing.assert_array_equal(tree_mask.numpy(), ref_tree_mask)


if __name__ == "__main__":
    unittest.main()
