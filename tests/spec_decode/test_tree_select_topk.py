"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.spec_decode.draft_tree_utils import tree_select_top_k


class TestDraftTreeUtils(unittest.TestCase):
    def setUp(self):
        self.tree_topk = 4
        self.hidden_size = 2048

    def test_tree_select_top_k_depth0(self):
        depth = 0
        topk_probs = paddle.to_tensor([[0.9, 0.1, 0.1, 0.1]], dtype="float32")
        topk_token_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        path_scores = None
        hidden_states = paddle.randn([1, self.hidden_size], dtype="float32")

        topk_token_ids, path_scores, tree_info, target_hidden_states = tree_select_top_k(
            depth, topk_probs, topk_token_ids, path_scores, hidden_states, self.tree_topk
        )

        ref_topk_token_ids = np.array([[1, 2, 3, 4]], dtype="int64")
        ref_path_scores = np.array([[0.9, 0.1, 0.1, 0.1]], dtype="float32")
        ref_tree_info = (
            np.array([[[0.9, 0.1, 0.1, 0.1]]], dtype="float32"),
            np.array([[1, 2, 3, 4]], dtype="int64"),
            np.array([[-1, 0, 1, 2, 3]], dtype="int64"),
        )
        np.testing.assert_array_equal(topk_token_ids.numpy(), ref_topk_token_ids)
        np.testing.assert_array_equal(path_scores.numpy(), ref_path_scores)
        np.testing.assert_array_equal(tree_info[0].numpy(), ref_tree_info[0])
        np.testing.assert_array_equal(tree_info[1].numpy(), ref_tree_info[1])
        np.testing.assert_array_equal(tree_info[2].numpy(), ref_tree_info[2])
        for hidden_states_per_token in target_hidden_states:
            np.testing.assert_array_equal(hidden_states_per_token.unsqueeze(0).numpy(), hidden_states.numpy())

    def test_tree_select_top_k_depth1(self):
        depth = 1
        topk_probs = paddle.to_tensor(
            [[0.9, 0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.6, 0.1, 0.1, 0.1]], dtype="float32"
        )
        topk_token_ids = paddle.to_tensor(
            [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]], dtype="int64"
        )
        path_scores = paddle.to_tensor([[0.9, 0.8, 0.7, 0.6]], dtype="float32")
        hidden_states = paddle.randn([self.tree_topk, self.hidden_size], dtype="float32")

        topk_token_ids, path_scores, tree_info, target_hidden_states = tree_select_top_k(
            depth, topk_probs, topk_token_ids, path_scores, hidden_states, self.tree_topk
        )

        ref_topk_token_ids = np.array([[11, 21, 31, 41]], dtype="int64")
        ref_path_scores = np.array([[0.9 * 0.9, 0.8 * 0.8, 0.7 * 0.7, 0.6 * 0.6]], dtype="float32")
        ref_tree_info = (
            np.array(
                [
                    [
                        [0.9 * 0.9, 0.9 * 0.1, 0.9 * 0.1, 0.9 * 0.1],
                        [0.8 * 0.8, 0.8 * 0.1, 0.8 * 0.1, 0.8 * 0.1],
                        [0.7 * 0.7, 0.7 * 0.1, 0.7 * 0.1, 0.7 * 0.1],
                        [0.6 * 0.6, 0.6 * 0.1, 0.6 * 0.1, 0.6 * 0.1],
                    ]
                ],
                dtype="float32",
            ),
            np.array([[11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44]], dtype="int64"),
            np.array([[4, 8, 12, 16]], dtype="int64"),
        )
        ref_selected_idx = np.array([0, 1, 2, 3], dtype="int64")
        np.testing.assert_array_equal(topk_token_ids.numpy(), ref_topk_token_ids)
        np.testing.assert_allclose(path_scores.numpy(), ref_path_scores)
        np.testing.assert_allclose(tree_info[0].numpy(), ref_tree_info[0])
        np.testing.assert_array_equal(tree_info[1].numpy(), ref_tree_info[1])
        np.testing.assert_array_equal(tree_info[2].numpy(), ref_tree_info[2])
        for i in range(self.tree_topk):
            np.testing.assert_array_equal(target_hidden_states[i].numpy(), hidden_states[ref_selected_idx[i]].numpy())

    def test_tree_select_top_k_depth0_batch2(self):
        depth = 0
        batch_size = 2
        topk_probs = paddle.to_tensor([[0.9, 0.1, 0.1, 0.1], [0.8, 0.2, 0.1, 0.1]], dtype="float32")
        topk_token_ids = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64")
        path_scores = None
        hidden_states = paddle.randn([batch_size, self.hidden_size], dtype="float32")

        topk_token_ids, path_scores, tree_info, target_hidden_states = tree_select_top_k(
            depth, topk_probs, topk_token_ids, path_scores, hidden_states, self.tree_topk
        )

        ref_topk_token_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64")
        ref_path_scores = np.array([[0.9, 0.1, 0.1, 0.1], [0.8, 0.2, 0.1, 0.1]], dtype="float32")
        ref_tree_info = (
            np.array([[[0.9, 0.1, 0.1, 0.1]], [[0.8, 0.2, 0.1, 0.1]]], dtype="float32"),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64"),
            np.array([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype="int64"),
        )
        np.testing.assert_array_equal(topk_token_ids.numpy(), ref_topk_token_ids)
        np.testing.assert_array_equal(path_scores.numpy(), ref_path_scores)
        np.testing.assert_array_equal(tree_info[0].numpy(), ref_tree_info[0])
        np.testing.assert_array_equal(tree_info[1].numpy(), ref_tree_info[1])
        np.testing.assert_array_equal(tree_info[2].numpy(), ref_tree_info[2])
        for i in range(batch_size):
            for j in range(self.tree_topk):
                np.testing.assert_array_equal(
                    target_hidden_states[i * self.tree_topk + j].numpy(), hidden_states[i].numpy()
                )

    def test_tree_select_top_k_depth1_batch2(self):
        depth = 1
        batch_size = 2
        topk_probs = paddle.to_tensor(
            [
                [0.9, 0.1, 0.1, 0.1],
                [0.8, 0.1, 0.1, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.6, 0.1, 0.1, 0.1],
                [0.5, 0.1, 0.1, 0.1],
                [0.4, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1, 0.1],
            ],
            dtype="float32",
        )
        topk_token_ids = paddle.to_tensor(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
                [51, 52, 53, 54],
                [61, 62, 63, 64],
                [71, 72, 73, 74],
                [81, 82, 83, 84],
            ],
            dtype="int64",
        )
        path_scores = paddle.to_tensor([[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]], dtype="float32")
        hidden_states = paddle.randn([batch_size * self.tree_topk, self.hidden_size], dtype="float32")

        topk_token_ids, path_scores, tree_info, target_hidden_states = tree_select_top_k(
            depth, topk_probs, topk_token_ids, path_scores, hidden_states, self.tree_topk
        )

        ref_topk_token_ids = np.array([[11, 21, 31, 41], [51, 61, 71, 52]], dtype="int64")
        ref_path_scores = np.array([[0.81, 0.64, 0.49, 0.36], [0.25, 0.16, 0.09, 0.05]], dtype="float32")
        ref_tree_info = (
            np.array(
                [
                    [
                        [0.9 * 0.9, 0.9 * 0.1, 0.9 * 0.1, 0.9 * 0.1],
                        [0.8 * 0.8, 0.8 * 0.1, 0.8 * 0.1, 0.8 * 0.1],
                        [0.7 * 0.7, 0.7 * 0.1, 0.7 * 0.1, 0.7 * 0.1],
                        [0.6 * 0.6, 0.6 * 0.1, 0.6 * 0.1, 0.6 * 0.1],
                    ],
                    [
                        [0.5 * 0.5, 0.5 * 0.1, 0.5 * 0.1, 0.5 * 0.1],
                        [0.4 * 0.4, 0.4 * 0.1, 0.4 * 0.1, 0.4 * 0.1],
                        [0.3 * 0.3, 0.3 * 0.1, 0.3 * 0.1, 0.3 * 0.1],
                        [0.2 * 0.2, 0.2 * 0.1, 0.2 * 0.1, 0.2 * 0.1],
                    ],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44],
                    [51, 52, 53, 54, 61, 62, 63, 64, 71, 72, 73, 74, 81, 82, 83, 84],
                ],
                dtype="int64",
            ),
            np.array([[4, 8, 12, 16], [4, 8, 12, 5]], dtype="int64"),
        )
        ref_selected_idx = np.array([[0, 1, 2, 3], [0, 1, 2, 0]], dtype="int64")
        np.testing.assert_array_equal(topk_token_ids.numpy(), ref_topk_token_ids)
        np.testing.assert_allclose(path_scores.numpy(), ref_path_scores)
        np.testing.assert_allclose(tree_info[0].numpy(), ref_tree_info[0])
        np.testing.assert_array_equal(tree_info[1].numpy(), ref_tree_info[1])
        np.testing.assert_array_equal(tree_info[2].numpy(), ref_tree_info[2])

        for i in range(batch_size):
            for j in range(self.tree_topk):
                np.testing.assert_array_equal(
                    target_hidden_states[i * self.tree_topk + j].numpy(),
                    hidden_states[i * self.tree_topk + ref_selected_idx[i][j]].numpy(),
                )


if __name__ == "__main__":
    unittest.main()
