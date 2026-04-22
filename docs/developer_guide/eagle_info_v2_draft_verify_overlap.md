# EAGLE Info V2: draft/verify overlap implementation note

This note summarizes how `eagle_info_v2` is wired into speculative decoding and how draft generation overlaps with target-model verify.

## 1) Main code paths

- Spec input mixins (v2 logic):
  - `python/sglang/srt/speculative/eagle_info_v2.py`
    - `EagleDraftInputV2Mixin.prepare_for_v2_draft`
    - `EagleDraftInputV2Mixin.prepare_for_extend_to_fill_draft_kvcache`
    - `EagleVerifyInputV2Mixin.prepare_for_v2_verify`
    - `EagleVerifyInputV2Mixin.sample`
- Worker orchestration:
  - `python/sglang/srt/speculative/eagle_worker_v2.py`
    - `EAGLEWorkerV2.forward_batch_generation`
    - `EAGLEWorkerV2.verify`
    - `EagleDraftWorker.draft`
    - `EagleDraftWorker._draft_extend_for_decode`
  - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`
    - same v2 overlap pattern for multi-layer draft model
- Stream toggle:
  - `python/sglang/srt/environ.py`
    - `SGLANG_ENABLE_OVERLAP_PLAN_STREAM`

## 2) Overlap principle (draft vs verify)

V2 uses a **plan stream + compute stream** split:

1. Draft worker produces speculative tree/token candidates (`draft()`).
2. Verify stage schedules `prepare_for_v2_verify(...)` inside `plan_stream_ctx`.
3. Main stream waits on plan stream (`current_stream().wait_stream(plan_stream)`), then runs target verify forward.
4. After verify sampling, draft-extend (`_draft_extend_for_decode`) prepares its next forward batch on the plan stream, then executes draft-extend compute on the main stream.

So the overlap is mainly **metadata/planning overlap** (batch/attn/KV planning in a side stream), while heavy model compute still runs on the main stream.

## 3) Scheduling, buffering, and synchronization

- Stream creation:
  - `_get_plan_stream()` in v2 workers creates a dedicated device stream when `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`.
- Sync points:
  - `wait_stream(...)` ensures plan-stream writes are visible before compute uses them.
- Buffer patch step:
  - In `verify()`, after waiting for plan stream, code calls
    `attn_backend.update_verify_buffers_to_fill_after_draft(...)`.
  - Rationale: some verify buffers (e.g. tree mask / positions) depend on draft output and may need post-draft correction.

## 4) KV cache handling

`eagle_info_v2.py` performs explicit cache-location planning:

- `prepare_for_v2_draft`:
  - allocates/assigns `batch.out_cache_loc` for draft tokens via Triton kernel
    `assign_draft_cache_locs_page_size_1`.
- `prepare_for_v2_verify`:
  - sets verify `batch.input_ids = draft_token`;
  - computes contiguous verify cache locations with `assign_extend_cache_locs_func`
    (Triton kernel `assign_extend_cache_locs` on CUDA/HIP).
- `prepare_for_extend_to_fill_draft_kvcache`:
  - updates `seq_lens`/`extend_seq_lens` and sets `ForwardMode.DRAFT_EXTEND_V2`,
    then builds a forward batch to materialize draft KV for the next iteration.

## 5) CUDA graph / piecewise graph usage

Yes, v2 uses a **piecewise CUDA graph style**:

- Draft-step graph runner:
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
  - captures/replays draft forward for supported batch-size buckets.
- Draft-extend graph runner:
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`
  - captures/replays extend path (`DRAFT_EXTEND_V2`) in separate graph buckets.
- Verify (target model) graph path:
  - `prepare_for_v2_verify` triggers target graph `replay_prepare(...)` if target graph can run.

This is not one monolithic graph across the full speculative control flow; instead it combines:
- graph replay for stable compute segments (draft / draft-extend / target verify),
- runtime stream scheduling + synchronization for dynamic control decisions.

## 6) End-to-end control/data flow (decode iteration)

1. `EAGLEWorkerV2.forward_batch_generation` (decode path):
   - `draft_worker.draft(...)` -> `EagleVerifyInput`.
2. `EAGLEWorkerV2.verify(...)`:
   - plan-stream `prepare_for_v2_verify(...)`;
   - sync + optional verify-buffer update;
   - target verify forward + acceptance sampling (`sample(...)`);
   - emits accepted tokens and next `EagleDraftInput`.
3. `EagleDraftWorker._draft_extend_for_decode(...)`:
   - plan-stream prepare extend batch;
   - sync;
   - run draft-extend compute (cuda graph if eligible);
   - prune/select states for next draft step.

This loop repeats per decode iteration.
