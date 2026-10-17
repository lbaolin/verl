# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | single   | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`
- `distillation.teacher_rollout_probability=0.0` (default: student generation; `1.0`: teacher generation; intermediate values choose once per global batch)

## Teacher-generated responses

The native synchronous V1 trainer can reuse the frozen vLLM teacher for response generation, then score the complete response with that teacher for supervised soft-target distillation:

```bash
DISTILLATION_LOSS_MODE=forward_kl_topk USE_POLICY_GRADIENT=False \
  bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
  distillation.teacher_rollout_probability=1.0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  distillation.distillation_loss.loss_max_clamp=null \
  distillation.distillation_loss.log_prob_min_clamp=null
```

Change only the probability to `0.5` for batch mixing or `0.0` for a student baseline. Validation always uses the student. Student and teacher must have matching tokenizers, token IDs, vocabulary, and EOS semantics. Teacher generation uses the student's sampling controls and full response budget, followed by a separate teacher scoring pass; it therefore adds teacher decoding cost.

Initial support is text-only single-turn generation with one teacher, vLLM, and supervised `forward_kl_topk` at temperature `1.0`. Policy-gradient/task-reward combinations, rollout logprobs/correction, auxiliary KL/entropy, custom agent loops/samplers, group filtering, multimodal inputs, speculative decoding, and async training are rejected when opted in. See [the OPD guide](../../docs/algo/opd.md#teacher-generated-training-rollouts) for source determinism and metrics. The launcher's default model and GPU sizes are examples, not a CPU smoke test.
