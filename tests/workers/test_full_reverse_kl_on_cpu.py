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

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.distillation import uses_local_teacher
from verl.trainer.distillation.fsdp.losses import compute_full_reverse_kl
from verl.trainer.ppo.utils import need_local_teacher_policy, need_teacher_policy, validate_local_teacher_policy
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationConfig, DistillationLossConfig, DistillationTeacherModelConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _reference_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    return (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)


@pytest.mark.parametrize("chunk_size", [1, 2, 64])
def test_full_reverse_kl_matches_reference_and_gradient(chunk_size):
    torch.manual_seed(42)
    student = torch.randn(2, 3, 17, requires_grad=True)
    teacher = torch.randn(2, 3, 17)
    reference_student = student.detach().clone().requires_grad_(True)

    actual = compute_full_reverse_kl(student, teacher, chunk_size=chunk_size)
    expected = _reference_reverse_kl(reference_student, teacher)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(student.grad, reference_student.grad, atol=1e-6, rtol=1e-6)


def test_full_reverse_kl_is_zero_for_identical_distributions():
    torch.manual_seed(7)
    student = torch.randn(4, 11, requires_grad=True)
    teacher = student.detach().clone()

    loss = compute_full_reverse_kl(student, teacher, chunk_size=2)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    loss.sum().backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad), atol=1e-7, rtol=0)


def test_full_reverse_kl_validates_shape_and_chunk_size():
    logits = torch.randn(2, 3)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_full_reverse_kl(logits, torch.randn(2, 4))
    with pytest.raises(ValueError, match="must be positive"):
        compute_full_reverse_kl(logits, logits, chunk_size=0)


def test_full_reverse_kl_empty_partition_keeps_student_grad_edge():
    student = torch.empty(0, 7, requires_grad=True)
    teacher = torch.empty_like(student)

    loss = compute_full_reverse_kl(student, teacher, chunk_size=2)

    assert loss.shape == (0,)
    assert loss.requires_grad
    loss.sum().backward()
    assert student.grad is not None
    assert student.grad.shape == student.shape


def test_full_reverse_kl_config_requires_direct_backpropagation():
    with pytest.raises(ValueError, match="use_policy_gradient=False"):
        DistillationLossConfig(loss_mode="full_reverse_kl", use_policy_gradient=True)


def test_full_reverse_kl_uses_colocated_teacher_without_remote_pool():
    loss_config = DistillationLossConfig(loss_mode="full_reverse_kl", use_policy_gradient=False)
    config = DistillationConfig(
        enabled=True,
        distillation_loss=loss_config,
        teacher_models={"teacher_model": DistillationTeacherModelConfig(model_path="teacher")},
    )
    assert uses_local_teacher(config)
    assert list(config.teacher_models) == ["default"]

    raw_config = OmegaConf.create(
        {"distillation": {"enabled": True, "distillation_loss": {"loss_mode": "full_reverse_kl"}}}
    )
    assert need_local_teacher_policy(raw_config)
    assert not need_teacher_policy(raw_config)


def test_local_teacher_rejects_reference_kl_settings():
    config = OmegaConf.create(
        {
            "algorithm": {"use_kl_in_reward": True},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            "distillation": {"enabled": True, "distillation_loss": {"loss_mode": "full_reverse_kl"}},
        }
    )
    with pytest.raises(ValueError, match="colocated reference engine"):
        validate_local_teacher_policy(config)


def test_local_teacher_device_residency_can_be_finalized_by_worker():
    config = FSDPEngineConfig()
    config.keep_forward_only_model_on_device = True
    assert config.keep_forward_only_model_on_device


@pytest.mark.parametrize("sequence_lens", [(3, 5), (5, 5)])
def test_vlm_position_ids_can_be_reindexed_after_transfer_queue_serialization(sequence_lens):
    position_ids = tu.nested_tensor_from_tensor_list(
        [torch.arange(length).expand(4, length) for length in sequence_lens], ragged_idx=2
    )
    if sequence_lens[0] == sequence_lens[1]:
        position_ids = torch.nested.as_nested_tensor(list(position_ids.unbind()), layout=torch.jagged)
    else:
        position_ids._ragged_idx = 1
    data = TensorDict({"position_ids": position_ids}, batch_size=[2])

    tu.maybe_fix_3d_position_ids(data)
    selected = tu.index_select_tensor_dict(data, [1])

    assert selected["position_ids"]._ragged_idx == 2
    expected = torch.arange(sequence_lens[1]).expand(4, sequence_lens[1])
    torch.testing.assert_close(selected["position_ids"].values(), expected)


def test_response_logit_mask_applies_causal_shift():
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([10, 11, 30, 31]), torch.tensor([20, 21, 22, 40])], layout=torch.jagged
    )
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": torch.tensor([[0, 10, 11], [20, 21, 22]]),
            "responses": torch.tensor([[30, 31, 0], [40, 0, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        },
        batch_size=[2],
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False

    actual = engine._response_logit_mask(data)
    expected = torch.tensor([[False, True, True, False, False, False, True, False]])
    torch.testing.assert_close(actual, expected)
