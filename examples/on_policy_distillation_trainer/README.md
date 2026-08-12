# On-Policy Distillation

This trainer optimizes a student on its own rollouts against a frozen teacher.
Sampled-token and top-k modes serve the teacher from a separate Ray pool;
`full_reverse_kl` colocates a frozen FSDP teacher with the actor and directly
backpropagates exact full-vocabulary reverse KL.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | local 8B | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, full_reverse_kl, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`

The Qwen3-VL script matches the paper's known Standard OPD setup: 8B teacher,
2B student, 16 prompts per batch, four student rollouts per prompt, five
epochs, per-sequence token means, and avg@8 validation. Learning rate and other
optimizer details, training-rollout sampling parameters, and sequence-length
limits were not reported by the paper and remain explicit script defaults.
