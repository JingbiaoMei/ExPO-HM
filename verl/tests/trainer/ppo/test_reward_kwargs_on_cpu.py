# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from omegaconf import OmegaConf

import verl.experimental.reward_loop.reward_manager as reward_manager_registry
from verl.trainer.ppo import reward as reward_utils


class _DummyRewardManager:
    def __init__(self, config, tokenizer, compute_score, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.kwargs = kwargs


def test_load_reward_manager_forwards_reward_model_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        captured_kwargs.update(kwargs)
        return 0.0

    monkeypatch.setattr(reward_utils, "default_compute_score", fake_default_compute_score)
    monkeypatch.setattr(reward_manager_registry, "get_reward_manager_cls", lambda _: _DummyRewardManager)

    config = OmegaConf.create(
        {
            "reward_model": {
                "reward_manager": {"source": "register", "name": "naive"},
                "reward_kwargs": {
                    "cde_weight": 0.2,
                    "low_entropy_cutoff": 0.1,
                },
            },
            "custom_reward_function": {"path": None},
        }
    )

    reward_manager = reward_utils.load_reward_manager(config=config, tokenizer=object())
    reward_manager.compute_score(
        data_source="meme_cde",
        solution_str="dummy solution",
        ground_truth="dummy gt",
        extra_info={},
    )

    assert captured_kwargs["cde_weight"] == 0.2
    assert captured_kwargs["low_entropy_cutoff"] == 0.1


def test_load_reward_manager_keeps_constructor_kwargs(monkeypatch):
    monkeypatch.setattr(reward_utils, "default_compute_score", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(reward_manager_registry, "get_reward_manager_cls", lambda _: _DummyRewardManager)

    config = OmegaConf.create(
        {
            "reward_model": {
                "reward_manager": {"source": "register", "name": "naive"},
                "reward_kwargs": {"cde_weight": 0.2},
            },
            "custom_reward_function": {"path": None},
        }
    )

    reward_manager = reward_utils.load_reward_manager(
        config=config,
        tokenizer=object(),
        reward_router_address="127.0.0.1:30000",
    )

    assert reward_manager.kwargs["reward_router_address"] == "127.0.0.1:30000"
