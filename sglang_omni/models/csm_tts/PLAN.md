# CSM-1B on sglang-omni — Implementation Plan

Repo: `/Users/venkat/Documents/p/venkat-github/sglang-omni`, branch `feat/csm-tts` @ aabc487.
Template: `sglang_omni/models/higgs_tts/` (canonical, re-verified in `/tmp/csm_sgl/contract.md`).
Model ground truth: `/tmp/csm_sgl/csm_spec.md` (HF transformers @ d557ef5, `sesame/csm-1b`).

Architecture (decided): new family `sglang_omni/models/csm_tts/`, 4-stage pipeline
`preprocessing → audio_encoder (Mimi encode, optional fast-path no-op) → tts_engine → vocoder (Mimi decode)`.
Scheduler-visible AR step = **one 80 ms frame**: paged backbone forward (1 new KV position) →
cb0 sample from `codebook0_head` → 31-step depth-decoder inner AR (dense static KV, 33 positions,
reset per frame, batched across the running batch) → 32-code frame → streamed to vocoder.
Backbone runs under SGLang paged/varlen attention (composed `LlamaForCausalLM`) — no dense
left-padded batch, which sidesteps the known bf16 B≥2 left-pad KV-corruption trap. Depth decoder is
custom dense torch, eager first, CUDA-graphable in M4.

Key CSM constants used throughout:
- backbone: 16L, d=2048, 32 heads/hd 64, 8 KV heads, ctx **2048**, rope θ=500000 llama3-scaled (factor 32, orig 1024)
- depth decoder: 4L, d=1024, 8 heads/hd **128** (explicit), 2 KV heads, ctx **33**, projector 2048→1024, head `W[31,1024,2051]`
- vocab: text 128256; audio 2051 (valid Mimi 0..2047; 2048/2049/2050 must NEVER reach Mimi); `codebook_eos=0`, `codebook_pad=2050`
- specials: bos 128000, eot 128001, `<|AUDIO|>`=128002, `<|audio_eos|>`=128003
- Mimi: 24 kHz, **12.5 Hz** ⇒ frame = 80 ms = **1920 samples**, 32 quantizers, codebook 2048
- frame embed = `Σ_{k=0..31} embed_audio_tokens[c_k + k*2051]` over ONE (65632, 2048) table (tied to depth embed)
- EOS: frame codebooks **0..30 all == 0** (cb31 excluded); audio trim at first **all-32-zero** frame; default cap 125 frames
- HF-default sampling: cb0 T=0.9 top-k 50; depth T=0.9 top-k 50 (separate param set)

---

## 1. File-by-file plan (dependency order)

All new files under `sglang_omni/models/csm_tts/` unless noted. Higgs analog in parens with its LOC
for calibration. Imports in `__init__.py` + `config.py` MUST stay sglang/CUDA-free (registry-scan
gotcha, contract §1).

### 1.1 `hf_config.py` (~110 LOC; higgs 66)
- `build_backbone_llama_config(csm_cfg) -> transformers.LlamaConfig` — synthesizes the SGLang-loadable
  backbone config from the **flat** HF `CsmConfig` backbone fields: hidden 2048, 16 layers, 32 heads,
  8 KV heads, head_dim 64, intermediate 8192, `max_position_embeddings=2048`, `rope_theta=500000`,
  `rope_scaling={"rope_type":"llama3","factor":32.0,"low_freq_factor":0.125,"high_freq_factor":0.5,
  "original_max_position_embeddings":1024}`, `vocab_size=128256`, **`tie_word_embeddings=True`**
  (so `LlamaForCausalLM`'s lm_head aliases embed_tokens — avoids a dead **525 MB** bf16 text head;
  we never use the backbone's lm_head).
- `build_depth_decoder_config(csm_cfg) -> SimpleNamespace/PretrainedConfig` — 4L/1024/8h/hd128/2kv,
  `max_position_embeddings=33`, depth rope (θ=500000, llama3 factor 32, low 0.001953125, high 0.0078125, orig 16).
- `class CsmTtsHfConfig(transformers.CsmConfig)`: overrides `get_text_config()` to return the realized
  synthetic LlamaConfig (SGLang `ModelConfig.from_server_args` reads head counts/hidden/layers/vocab
  through it; native `CsmConfig.get_text_config()` would return self with `vocab_size=2051`).
  Subclassing the native class keeps `isinstance` and field compatibility for HF-side tooling.

### 1.2 `__init__.py` (~20 LOC; higgs 20)
- `AutoConfig.register("csm", CsmTtsHfConfig, exist_ok=True)` (overrides the native mapping in-process;
  risk + mitigation in §8.R7), then `from . import config`. Nothing else.
  Semantics: `register(exist_ok=True)` writes to `_LazyConfigMapping._extra_content`, which
  `__getitem__` consults BEFORE the native mapping → the override wins in-process; the model-type
  consistency check passes because our subclass inherits `model_type="csm"` from `CsmConfig`. This
  is stable transformers behavior across the pinned range (`transformers<5.0`), but it is not
  importable on the dev box — **confirm in the container at M1** (one-liner:
  `AutoConfig.for_model("csm")` is our class after import).

### 1.3 `utils.py` (~140 LOC; higgs 158)
- Constants: `NUM_CODEBOOKS=32`, `CODEBOOK_VOCAB=2051`, `MIMI_CODEBOOK_SIZE=2048`, `CODEBOOK_EOS=0`,
  `CODEBOOK_PAD=2050`, `STOP_CODE=-1`, `AUDIO_TOKEN_ID=128002`, `AUDIO_EOS_TOKEN_ID=128003`,
  `BOS=128000`, `EOT=128001`, `SAMPLE_RATE=24000`, `SAMPLES_PER_FRAME=1920`, `FRAME_RATE=12.5`,
  `DEFAULT_MAX_FRAMES=125`, `BACKBONE_CTX=2048`, `K_MAX=2051`.
- `resolve_checkpoint(model_path)` (HF snapshot, copy of higgs utils.py:82-86).
- `get_or_load_codec(checkpoint_dir, device, dtype) -> CsmMimiCodec` process-wide cache
  (one Mimi load serves audio_encoder + vocoder; ~0.4 GB fp32 saved per dedup).
- `encoded_audio_length(num_samples) -> int` — closed-form port of HF `processing_csm.py:93-129`
  (causal-conv ladder ceil; ≈ `ceil(S/1920)`). Used ONLY for pre-encode budget/admission estimates on
  raw waveforms (the pre-encoded fast path needs no formula — F is just `codes.shape[0]`); the
  raw-waveform path defers the actual placeholder count to the real encode (§1.14) so prompts can
  never mismatch. Do NOT "optimize" the deferral away by promoting this estimate to prompt assembly.
- `to_codes_F32(obj)` coercion for client-supplied pre-encoded context codes `[F,32]`.
- Audio loaders (path/URL/bytes/base64 → 24 kHz mono float32 `[1,1,S]`), copied from higgs.
- **No delay-pattern helpers** — CSM has none.

### 1.4 `modeling.py` (~260 LOC; higgs 58 — this is the big delta: CSM has a real inner transformer)
Framework-free torch. Classes:
- `CsmFrameEmbedding(nn.Module)` — one `weight [65632, 2048]`.
  - `forward(codes_B32) -> [B, 2048]`: `F.embedding(codes + offsets).sum(-2)` with registered
    `audio_tokens_offsets = arange(32)*2051` buffer (mirrors HF modeling_csm.py:655-666).
  - `embed_codebook(codes_B, k) -> [B, 2048]`: single lookup at offset `k*2051` (depth-step feedback).
- `CsmCodebooksHead(nn.Module)` — `weight [31, 1024, 2051]`, stored EXACTLY in checkpoint layout;
  `forward(h_BD, step) -> [B, 2051]` = **`h @ weight[step]`** (≡ `F.linear(h, weight[step].T)`,
  exactly HF modeling_csm.py:528-536; head *k* predicts codebook *k+1*). ⚠ Trap: do NOT "fix" shapes
  by re-storing the weight as `[31, 2051, 1024]` — combined with loading the checkpoint tensor
  `(31,1024,2051)` untransposed that RUNS and produces garbage audio. The §6.1 tiny-config depth
  parity test (mandatory before M1 exit) pins the orientation.
- `CsmDepthDecoderLayer(nn.Module)` — RMSNorm → SDPA attn (8h, hd 128, 2 KV heads GQA, llama3 rope) →
  RMSNorm → SwiGLU(1024→8192→1024). Attention computed over the **fixed 33-length** static KV with an
  additive causal/length mask — shape-static (CG-ready), and at 33 positions the masked-full compute
  is cheaper than any varlen cleverness.
- `CsmDepthDecoder(nn.Module)` — owns `inputs_embeds_projector [1024,2048]`, 4 layers, final norm,
  `codebooks_head`, precomputed 33-position rope cos/sin, and **slot-indexed static KV**
  `k,v: [4, max_slots, 2, 33, 128]` bf16. "Reset per frame" = reset a length scalar / mask, no memset.
  - `generate_frame(h_B2048, cb0_B, temps_B, top_ks_B, *, bs) -> codes_B32`:
    step 0+1 fwd of `[proj(h), proj(embed_codebook(cb0, 0))]` (len-2 "depth prefill") → head `W[0]` →
    sample cb1 → 30 more len-1 fwd steps: pos *p* embeds prev code with table offset `(p-1)*2051`,
    head `W[p-1]` samples cb_p — total **31 sampled codes** (cb31 never forwarded; HF parity).
    Fixed 31-iteration host loop, no data-dependent branches ⇒ capturable as one graph in M4.
    Logits fp32 before sampling; per-row branchless greedy/top-k via `sampler.sample_codes_batched`.

### 1.5 `sampler.py` (~300 LOC; higgs 397 — simpler state machine, dual param sets)
- `CsmBatchedSamplerState` — pool `[P]`: `last_codes int64 [P,32]`, `generation_done bool [P]`,
  `frames_emitted int32 [P]`; `reset_row(row)`. (No `delay_count`/`eoc_countdown` — CSM has no delay
  pattern and no wind-down.)
- `sample_codes_batched(logits_BV_fp32, temperature_B, top_k_buf_B, top_p_B=None) -> [B]` —
  copy of higgs `_sample_independent_batched` with `K_MAX=2051`: greedy rows (T≤1e-5 or top_k==1)
  short-circuit branchlessly to argmax over RAW logits; fixed-shape `topk(K_MAX)` + per-row k-th-value
  gather. Used for BOTH cb0 and every depth step (CG-deterministic at temp=0).
- `frame_finalize_direct(codes_B32, generation_done_B) -> (out_codes_B32, new_done_B, was_done_B)` —
  the branchless EOS machine: `eos_now = (codes[:, :31] == 0).all(-1)` (cb31 excluded, HF-exact);
  `new_done = done | eos_now`; `out_codes = where(was_done, STOP_CODE, codes)`; done rows freeze.
- `step_reference(...)` — per-row eager mirror for parity tests (higgs pattern,
  tests assert per-row vs batched equality).

### 1.6 `weight_loader.py` (~90 LOC; higgs 54)
`CsmWeightMapper` remap of the HF shards (`transformers-0000*-of-00002.safetensors`, 538 tensors, fp32):
- `embed_text_tokens.weight` → `backbone.model.embed_tokens.weight` (text vocab IS the backbone
  embedding; lm_head tied to it, never used)
- `backbone_model.embed_tokens.embed_audio_tokens.weight` → `frame_embedding.weight`
- `backbone_model.layers.{i}.*` → `backbone.model.layers.{i}.*` (Llama qkv/gate_up stacking handled by
  `LlamaForCausalLM.load_weights`); `backbone_model.norm.weight` → `backbone.model.norm.weight`
- `lm_head.weight [2051,2048]` → `codebook0_head.weight` (**must not** hit the backbone lm_head)
- `depth_decoder.model.inputs_embeds_projector.weight` → `depth_decoder.inputs_embeds_projector.weight`
- `depth_decoder.model.layers.{0..3}.*` → `depth_decoder.layers.{i}.*` (own dense modules — manual
  q/k/v/o + gate/up/down copies, no stacking); `depth_decoder.model.norm.weight` → depth final norm
- `depth_decoder.codebooks_head.weight` → `codebooks_head.weight`
- `depth_decoder.model.embed_tokens.weight` → skip (tied alias of the audio table). It IS present in
  the 538-tensor census (verified vs `tf_index.json`), so the skip path ALWAYS fires for this
  artifact and the M1 zero-unexpected census must count it as deliberately-skipped, not missing
- `codec_model.*` → return None (skip in engine; `audio_codec.py` loads that subtree itself)
- everything cast to backbone dtype (bf16) on copy.

### 1.7 `model.py` (~520 LOC; higgs 553)
`class CsmTTSModel(nn.Module)` — registered in SGLang's ModelRegistry (shared edit §1.16).
- `__init__(self, config: CsmTtsHfConfig, quant_config=None, prefix="", max_batch_size=64)`:
  - `self.backbone = LlamaForCausalLM(build_backbone_llama_config(config), quant_config,
    prefix=add_prefix("backbone", prefix))` — paged attention/KV/CUDA-graph inherited.
    (Verified against upstream sglang `srt/models/llama.py`: ctor is exactly
    `(config, quant_config=None, prefix="")`; `tie_word_embeddings=True` makes `lm_head` a pure
    alias of `embed_tokens` and `load_weights` skips any incoming `lm_head.weight`; `rope_scaling`
    is passed through to `get_rope`. Cheap re-confirm against the pinned 0.5.8 wheel at M1.)
  - `self.frame_embedding = CsmFrameEmbedding(...)`, `self.codebook0_head = nn.Linear(2048, 2051,
    bias=False)`, `self.depth_decoder = CsmDepthDecoder(depth_cfg, frame_embedding, max_slots=pool_size)`.
  - Cast frame_embedding/codebook0_head/depth_decoder to backbone bf16 (higgs "~1 ULP/step" note).
  - Sampler pool `CsmBatchedSamplerState(pool_size = max_batch_size + 1)`, last row = padding row;
    `_rid_to_row` / `_free_rows` / `acquire_row` / `release_row` / `reset_request` — verbatim higgs.
  - CG shadow buffers (pool_size-sized, sliced `[:bs]` in-graph):
    `_cg_row_indices long`, `_cg_temperature/_cg_top_p fp32`, `_cg_top_k_buf long (=K_MAX)`,
    **`_cg_depth_temperature fp32`, `_cg_depth_top_k_buf long`** (CSM delta: second param set),
    `_cg_codes_BN long [P,32]`, `_cg_collect_staging long [P,34]`, `_cg_was_done bool`,
    `_cg_active_generation_done bool`, `_cg_active_last_codes long [P,32]`.
- `forward(input_ids, positions, forward_batch, input_embeds=None) -> LogitsProcessorOutput`:
  decode → `input_embeds = _decode_step_embeds_cg(bs)`; prefill → runner-supplied overlay embeds;
  `hidden = self.backbone.model(input_ids, positions, forward_batch, input_embeds)` (SGLang LlamaModel
  applies the final RMSNorm — verified at upstream `srt/models/llama.py:400`
  `hidden_states, _ = self.norm(hidden_states, residual)` — matching CSM's **post-norm** depth
  conditioning requirement exactly; HF side confirmed at output_capturing.py:205-267 +
  modeling_csm.py:747-751);
  last-token hidden via `cumsum(extend_seq_lens)-1` on prefill; then `decode_codebooks_batch_cg(hidden)`
  (decode) or `decode_codebooks_batch(hidden, req_ids, gen_params)` (prefill — **frame 0 is sampled at
  prefill**, like higgs); returns dummy `LogitsProcessorOutput(next_token_logits=zeros(bs, 128256, fp32))`.
- `_decode_step_embeds_cg(bs) -> [bs, 2048]`: **unconditional** `frame_embedding(_cg_active_last_codes[:bs])`
  — post-prefill a last frame always exists (no higgs `delay_count>0` text fallback; padding rows embed
  zeros-garbage that the collect discards). Simpler than higgs by design.
- `decode_codebooks_batch_cg(hidden_BD)`: `logits0 = codebook0_head(hidden).float()` →
  `cb0 = sample_codes_batched(logits0, _cg_temperature, _cg_top_k_buf)` →
  `codes = depth_decoder.generate_frame(hidden, cb0, _cg_depth_temperature, _cg_depth_top_k_buf, bs=bs)`
  → `frame_finalize_direct` → write `_cg_codes_BN/_cg_was_done/_cg_active_*`. No host control flow, no D2H.
- `decode_codebooks_batch(...)`: eager pool-indexed variant for prefill (gen params from req data,
  rows via `acquire_row`), appends to `_output_codes[rid]`. **Runs the SAME `frame_finalize_direct`
  EOS machine on frame 0** — a frame-0 EOS is legal HF behavior (the EOS check covers the first
  sampled frame, generation_csm.py:219-228 + 301-307) and empirically real for this checkpoint
  (rime #169: "model emits EOS at frame 0 → 1-chunk stream"). On `generation_done` at prefill:
  `_mark_sampler_finished` immediately, emit NOTHING to the vocoder, and let the zero-decode-step
  lifecycle run (KV release via finish reason; vocoder sees `on_stream_done` with zero chunks →
  empty/near-empty audio, matching HF `cutoff_idx=0`).
- `load_weights(weights_iter)`: split stream per §1.6; shape-check own params.

### 1.8 `payload_types.py` (~100 LOC; higgs 105)
`@dataclass CsmTtsState` + `to_dict/from_dict` (THE inter-stage wire schema): `text`, `speaker_id`,
`context: list[{speaker_id, text, codes_F32|waveform}]`, `prompt_ids`, `context_codes` (CPU int32
`[F,32]` per segment), `num_ctx_codes_consumed` (chunked-prefill cursor), gen params
(`max_new_tokens` frames, `temperature/top_k/top_p`, `depth_temperature/depth_top_k`, `seed`,
`stream`), `output_frames` (list of `[32]`), usage fields (`prompt_tokens`, `completion_frames`,
`engine_time_s`).

### 1.9 `text_tokenizer.py` (~90 LOC; higgs 72)
`CsmPromptBuilder` over the checkpoint's Llama-3 tokenizer (loaded from raw `tokenizer.json`,
higgs trick, transformers-v5 metadata dodge). Mirrors the hub chat template **exactly** (parity
depends on it): per context message `128000 ⊕ tok("[<spk>]<text>") ⊕ 128001 ⊕ 128002×F ⊕ 128003`;
final message `128000 ⊕ tok("[<spk>]<text>") ⊕ 128001`; `add_special_tokens=False`, speaker tag is
literal text `[0]`. Returns `prompt_ids` + per-segment placeholder spans. Note: the `<|AUDIO|>`
positions keep their REAL id 128002 (in-vocab) — unlike higgs's −100 sentinel, no masking needed;
the prefill overlay simply overwrites those embedding rows.

### 1.10 `request_builders.py` (~200 LOC; higgs 202)
- `build_sglang_csm_request(state, payload) -> CsmSGLangRequestData`: SGLang `SamplingParams`
  (max_new_tokens=frames, temperature, top_k, top_p, **`sampling_seed=int(seed)`** — the kwarg is
  `sampling_seed`, not `seed`; request_builders.py:77-78) + **`sampling_params.normalize(tokenizer=None)`**
  (mandatory — stop_strs crash otherwise); `Req(rid, origin_input_text="", origin_input_ids=prompt_ids,
  sampling_params, vocab_size=128256, extra_key=_context_fingerprint(state))` (`origin_input_text=""`
  required — positional signature, higgs request_builders.py:88-98); stamp
  `req._codec_suppress_tokens=None`, `req._input_embeds_are_projected=False` (V1 prefill-manager probes).
- `_context_fingerprint(state)`: blake2b-16 over all context-code matrices packed 2 B/code + speaker
  ids; `None` for context-free requests (shared radix subtree). Required because different context
  audios share identical `128002×F` token prefixes — without namespacing the radix tree would share
  KV across different voices (higgs lesson, verbatim).
- `build_csm_stream_metadata(state)`: `{modality:"audio_codes", stream:True, num_codebooks:32,
  codebook_size:2051, [initial_codec_chunk_frames]}`.
- `make_csm_scheduler_adapters(model, max_new_tokens_cap=125)`: request_builder stamps
  engine_start_s/stage_payload/stream_metadata + clamps `max_new_tokens` to
  **`min(cap, BACKBONE_CTX - 1 - len(prompt_ids))`** — the scheduler's budget is
  `max_req_len = min(context_length - 1, max_total_num_tokens - 1)` = **2047**, not 2048
  (omni_scheduler.py:150-153); clamping against 2048 admits requests the admission check then
  rejects; result_adapter writes `output_frames` + usage into state,
  calls `model.reset_request(rid)` (frees sampler row), returns fresh StagePayload.

### 1.11 `model_runner.py` (~430 LOC; higgs 412)
`class CsmTTSModelRunner(ModelRunner)` — full hook set, see §2 for design detail.
Methods: `before_prefill` (embed overlay), `before_decode → _populate_cg_buffers`,
`post_decode → _collect_step_outputs_cg`, `post_decode_launch` / `post_decode_resolve` (async halves),
`_decode_pack_gpu` (staging `[P, 34]`), `_decode_collect_host`, `_collect_step_outputs` (prefill/eager),
`_build_prefill_input_embeds`, `_extract_decode_sampling_params` (+ depth-param extraction from
`req._omni_data`), `_emit_code_chunk(target="vocoder")`, `_mark_sampler_finished`, `set_stream_outbox`.

### 1.12 `audio_codec.py` (~240 LOC; higgs 279)
`class CsmMimiCodec` — loads `MimiModel` from the SAME checkpoint's `codec_model.*` subtree
(one-artifact pattern). **fp32 by default** (conv-transpose decode stability; bf16 opt-in).
- `SAMPLE_RATE=24000`, `samples_per_frame=1920` (asserted from config `sampling_rate/frame_rate`).
- `encode_reference(wav_11S, sample_rate) -> codes_F32 long` (pads to ≥1 frame; transposes HF's `[1,32,F]`).
- `decode(codes_F32) -> wave_1S` (transpose to `[1,32,F]`, clamp guard, decode).
- `decode_batch(items) -> list` via `_bucketed_batch` (exact-frame-count buckets, higgs copy).
- `streaming_decode(codes_F32, past_key_values) -> (wave, past_key_values)` — wraps
  `MimiModel.decode(..., decoder_past_key_values=...)`; per-request state object owned by the vocoder
  scheduler (M4+ path; M2/M3 use stateless overlap-trim, §4).
- Hard clamp helper `sanitize_for_mimi(codes)`: `codes.clamp_(0, 2047)` after asserting/counting
  `(codes >= 2048).sum()` for the parity-gate guard metric.

### 1.13 `vocoder_scheduler.py` (~480 LOC; higgs 547)
`class CsmStreamingVocoderScheduler(StreamingSimpleScheduler)` — §4 has the full design.
ctor knobs: `stream_stride=13`, `stream_followup_stride=6`, `stream_overlap_frames=4`,
`stream_holdback_frames=2`, `max_batch_size=8`, `max_batch_wait_ms=2`.
Hooks: `is_streaming_payload`, `on_streaming_new_request` (latch num_codebooks=32/codebook_size=2051 +
`initial_codec_chunk_frames`), `on_stream_chunk` (append `[32]` row → `_decode_delta`),
`on_stream_done` (final flush + slim terminal result with usage), `clear_stream_state` (drops
per-request `CsmStreamState` incl. any Mimi `past_key_values` — the cross-request state-leak guard).

### 1.14 `stages.py` (~430 LOC; higgs 501)
Four factories:
- `create_preprocessing_executor(model_path, *, max_concurrency=8)` → `ThreadedSimpleScheduler`.
  Parses `payload.request.inputs` (str | dict with `input|text`, `speaker`, `context:[{text, speaker,
  audio|codes}]`, higgs-compatible `references` aliasing); three branches: (a) pre-encoded codes →
  full prompt now via `encoded` F; (b) no context → prompt now; (c) raw waveform → load/resample
  24 kHz mono, **defer prompt assembly to audio_encoder** (placeholder count must come from the actual
  encode — kills the off-by-one class). Limits: context audio ≤ 60 s (= 750 backbone positions; budget
  guard), text ≤ ~1500 tokens after reserving frames. Three-tier caching copied from higgs
  (path-hash memo / waveform LRU / encoded-codes LRU as CPU int32).
- `create_audio_encoder_executor(model_path, *, device, max_batch_size=8, max_batch_wait_ms=2)` →
  `SimpleScheduler`. No-op on fast path; else Mimi `encode_reference` → `[F,32]` → build prompt with
  exactly F placeholders + `128003` → null waveform. Startup warmup encode of 1 s zeros.
- `create_sglang_tts_engine_executor(model_path, *, device, max_new_tokens=125,
  server_args_overrides=None, enable_async_decode=False, async_decode_min_batch_size=2)` —
  the §3-contract recipe: `resolve_checkpoint` → `build_sglang_server_args(checkpoint_dir,
  context_length=2048, disable_cuda_graph=True (M1–M3; False in M4), cuda_graph_max_bs=8,
  mem_fraction_static=0.5, max_running_requests=8, chunked_prefill_size=2048, dtype="bfloat16",
  **overrides)` → `server_args.disable_overlap_schedule = True` → `create_sglang_infrastructure` →
  `CsmTTSModelRunner(model_worker, SGLangOutputProcessor(capture_hidden=False, ...))` →
  `make_csm_scheduler_adapters` → `OmniScheduler(..., abort_callback=model.reset_request)` →
  `model_runner.set_stream_outbox(scheduler.outbox)`. No `truncate_rope_to_bf16` (CSM ckpt is fp32;
  no bf16-training-parity rationale — leave SGLang's fp32 cos/sin cache alone).
- `create_vocoder_executor(model_path, *, device, dtype="float32", ...)` → `get_or_load_codec` →
  `CsmStreamingVocoderScheduler`.

### 1.15 `config.py` (~80 LOC; higgs 78)
`class CsmTtsPipelineConfig(PipelineConfig)`: `architecture: ClassVar = "CsmForConditionalGeneration"`;
4 `StageConfig`s, all `process="pipeline"`, GPU stages `gpu=0`; **every non-terminal stage MUST set
`next=`** (`preprocessing.next="audio_encoder"`, `audio_encoder.next="tts_engine"`,
`tts_engine.next="vocoder"`) — schema validation requires exactly one of `next`/`terminal` per stage
(`config/schema.py:298-302`, higgs sets it at config.py:34,45,60; omitting it fails at startup);
`tts_engine.stream_to=["vocoder"]`; vocoder `terminal=True, can_accept_stream_before_payload=True`.
`EntryClass = CsmTtsPipelineConfig`.
**No sglang imports; inline the concurrency constant** (do NOT import stages.py the way higgs
config.py:9 does — that import chain is the registry-silent-skip gotcha).

### 1.16 Shared-file edits
- `sglang_omni/model_runner/sglang_model_runner.py:87-98` — add
  `"CsmForConditionalGeneration": "sglang_omni.models.csm_tts.model:CsmTTSModel"` (+1 LOC; the only
  mandatory shared edit).
- (optional, M3) `sglang_omni/cli/serve.py:17-19` — generalize `_HIGGS_ASYNC_DECODE_FACTORY` into a
  set including `sglang_omni.models.csm_tts.stages.create_sglang_tts_engine_executor` so
  `--async-decode` works for csm (+5 LOC). Until then, dotted overrides work.

### 1.17 Tests + assets
- `tests/unit_test/csm_tts/{__init__.py, test_modeling.py (~250), test_sampler.py (~220),
  test_pipeline.py (~600), test_request_builders.py (~60), test_async_decode_runner.py (~250)}` — §6.
- `scripts/parity_csm_hf.py` (~200) — golden-parity harness (§6.2).
- `examples/csm_tts/csm-1b-rtx3060.yaml` (~40) — §5.

Total new code ≈ **3.1 kLOC** + ≈ **1.4 kLOC** tests.

---

## 2. Frame-step model_runner design

### 2.1 Where the depth loop lives
Inside `model.forward`'s decode branch via `decode_codebooks_batch_cg` — i.e. **inside the
scheduler-visible step, after the paged backbone forward, before `_finalize`**. The scheduler still
sees a normal one-token decode step (1 frame = 1 backbone position = 1 "token" of SGLang bookkeeping;
`max_new_tokens` counts frames, exactly like HF). The 31-step inner loop is a fixed-trip-count host
loop over static-shape tensors: legal eager, and capturable as one fused graph segment in M4 (the
loop unrolls at capture; no data-dependent host control flow anywhere in it).

Step anatomy (decode, sync path):
1. `before_decode → _populate_cg_buffers` (outside any captured region): reset padding row; acquire
   rows; write `_cg_row_indices[:bs]`; lookahead-only GPU-side done-row reroute to the padding row
   (higgs guard #1, verbatim); fill cb0 sampling buffers from SGLang `sampling_info` via
   `_flat_sampling_attr` (one D2H per attribute); **fill depth sampling buffers from
   `req._omni_data.data` host-side** (depth params never ride SGLang's sampling_info — they're
   CSM-private, defaulted to (0.9, 50) or mirrored from cb0 params when the request says so);
   gather `pool.{generation_done,last_codes}[rows] → _cg_active_*[:bs]`.
2. `model.forward`: `_decode_step_embeds_cg` (unconditional frame embed of `last_codes`) → paged
   backbone (1 new position, varlen, per-request KV pages) → post-norm hidden `[bs, 2048]` →
   `decode_codebooks_batch_cg`: cb0 head+sample → `depth_decoder.generate_frame` (31 inner steps over
   the slot-indexed `[4, cap, 2, 33, 128]` static KV, batched across `[:bs]`, reset-by-length-scalar
   per frame) → `frame_finalize_direct` EOS machine → `_cg_codes_BN[:bs]` written. Dummy text logits out.
3. `post_decode → _collect_step_outputs_cg`: `_decode_pack_gpu` (scatter `_cg_active_* → pool[rows]`;
   pack staging) → ONE blocking D2H `staging[:n].cpu()` → `_decode_collect_host`.

### 2.2 Depth KV is slot-indexed, NOT pool-indexed (key simplification vs higgs)
The depth cache is frame-local: reset at the start of every frame, dead by the end. It therefore
needs **no per-request persistence**, no rid→row indirection, no gather/scatter — it lives in
CG-batch-slot order `[:bs]` and is simply overwritten next step. Only the *sampler* state
(`last_codes`, `generation_done`, `frames_emitted`) needs the higgs pool-row discipline, because it
must survive across steps while batch composition changes under a fixed-shape graph.

### 2.3 Sampler-pool rows for CSM
Per row: `last_codes int64 [32]` (frame-embed feedback), `generation_done bool`,
`frames_emitted int32`. CG shadow set: `_cg_active_{generation_done,last_codes}`, plus param buffers
`_cg_temperature/_cg_top_p/_cg_top_k_buf` (cb0) and `_cg_depth_temperature/_cg_depth_top_k_buf`
(depth). Padding row reset every step; padding param values neutral (T=1, k=K_MAX).

### 2.4 Changes vs the Higgs runner (exhaustive)
| Higgs | CSM |
|---|---|
| `delay_count`/`eoc_countdown` in pool + shadow | gone — no delay pattern, no wind-down |
| `_decode_step_embeds_cg`: `where(delay_count>0, fused, text_embed)` | unconditional `frame_embedding(last_codes)` |
| fused parallel 8-codebook head, 1 GEMM | cb0 head + 31-step depth AR (own module, static KV) |
| one sampling param set | two: cb0 (from SGLang sampling_info) + depth (from `_omni_data`) |
| staging `[P, N+2] = [P, 10]` | `[P, 34]` = `c0..c31 | was_done | generation_done` |
| prefill overlay pastes at `-100` sentinel positions | pastes at real-id `128002` positions; `128003` position gets the all-zeros-frame embed (HF `_merge_input_ids_with_input_values` parity); `num_ctx_codes_consumed` cursor keeps it chunked-prefill-correct |
| `_mark_sampler_finished`: `FINISH_MATCHED_TOKEN(EOC_ID=1025)` | `FINISH_MATCHED_TOKEN(matched=CODEBOOK_EOS=0)` (cb0 of the EOS frame; upstream only needs *a* finish reason to fire KV release) |
| prefill samples step-0 codes via `decode_codebooks_batch` | same — **frame 0 is sampled at prefill** (cb0 + full depth loop on the last prompt position's hidden) and runs through `frame_finalize_direct`; non-EOS → streamed immediately; **frame-0 EOS → finished at prefill, nothing streamed, zero-decode lifecycle** (§1.7 — empirically real for this checkpoint); `next_token_ids` overwritten with cb0 |
| `truncate_rope_to_bf16` post-load | not applied (fp32-trained ckpt) |

### 2.5 EOS / wind-down → the branchless state machine
Higgs's three-phase machine (delay window → EOC countdown N−2 → done) collapses to one transition:
`eos_now = all(codes[:31] == 0)` computed inside `frame_finalize_direct`, `done |= eos_now`,
done rows emit `STOP_CODE=-1` and freeze. The **`was_done` flag and all three async overrun guards
stay verbatim** (padding-row reroute at launch; `pre_finished` snapshot + row drop in
`_resolve_and_process`; post-drain `filter_batch()`), because they protect the lookahead overlap, not
the wind-down. Emission rule: the EOS frame is appended to `data.output_frames` (HF `sequences`
include it — needed for exact parity comparison) but `_emit_code_chunk` **skips it** (`new_done` rows
never stream), so no stop frame can reach Mimi; the vocoder's ≥2048 clamp + all-zero-frame trim are
the second and third fences. The same finalize/emission rule applies to the prefill-sampled frame 0
(§1.7). **Documented deviation (streaming only):** HF's non-streaming decode trims at the first
all-32-zero frame, so an EOS frame with cb31≠0 IS decoded by HF (all its codes are valid 0..2047,
generation_csm.py:471-478) — our streamed audio is then one 80 ms frame shorter than the HF
reference. Our non-streaming path matches HF exactly (trim over `output_frames`, which include the
EOS frame). Consequence: exclude the final frame from any streamed-vs-HF waveform comparison (M2 ASR
sanity, M5 cross-engine integrity).

### 2.6 Packed D2H row + host collect
`_cg_collect_staging[:n] = [codes_0..31 | was_done | generation_done]` int64 — one D2H per step
(blocking `.cpu()` sync path; `copy_(non_blocking=True)` into ping-pong pinned buffers async path).
`_decode_collect_host` per row: skip if `req.is_chunked > 0` / `req.finished()` / `was_done`;
else append `codes_32` to `data.output_frames`; if `generation_done`: `_mark_sampler_finished`, no
stream emit; else `_emit_code_chunk(OutgoingMessage(type="stream", target="vocoder",
data=codes_32 CPU int64 [32], metadata=stream_metadata))`; collect cb0 into `result.next_token_ids`.
Async halves: `post_decode_launch` publishes `next_token_ids = _cg_codes_BN[:n,0].clamp_min(0)` from
GPU state (clamp keeps STOP=-1 in range; decode embeds read `last_codes`, so this is bookkeeping only);
`post_decode_resolve` runs the same host collect off the pinned snapshot. Async stays off until M4;
`async_decode_min_batch_size=2` kept (the measured bs=1 regression).

---

## 3. KV / memory plan

### 3.1 Paged backbone
- `context_length = 2048` (hard model ceiling), SGLang default token-granularity pages
  (page_size=1; nothing CSM-specific needed). Radix tree namespaced by `extra_key` context fingerprint.
- KV bytes/position: 16 layers × 2 (K+V) × 8 heads × 64 dim × 2 B (bf16) = **32 KiB/position**.
  Full-context request = **64 MiB**.
- Prompt cost examples: text ≈ 1 pos/token; context audio = `ceil(S/1920)` + 1 (`<|audio_eos|>`)
  ≈ **12.5 pos/s + 1**. Typical voice-prompted request: 5 s context (64 pos) + ~25 text tokens
  + 125 generated frames ≈ **215 positions ≈ 6.7 MiB KV**. Admission check
  (`input_len + max_new_tokens ≤ max_req_len`, omni_scheduler:567-589) enforces the budget
  pre-schedule — note `max_req_len = min(context_length - 1, max_total_num_tokens - 1)` = **2047**
  (omni_scheduler.py:150-153; the §1.10 clamp accounts for the −1) — with the actionable
  `mem_fraction_static` hint (whose wording is Qwen-flavored, `--thinker-mem-fraction-static`;
  cosmetic only).

### 3.2 Dense depth KV (per CG slot, not per request)
4 layers × 2 (K+V) × 2 KV heads × 33 pos × 128 dim × 2 B = **264 KiB/slot** →
8 slots ≈ **2.1 MiB** total. Noise.

### 3.3 RTX 3060 12 GB budget (bf16 engine, fp32 Mimi, all four stages colocated on gpu 0)

**This table is engineering arithmetic, NOT framework-enforced.** The colocation budget check
(`require_memory_fraction_for_colocation`, `config/topology.py:261-272`) only fires when a GPU is
shared by **more than one process group**; all four CSM stages run `process="pipeline"` → one
group → the check is skipped (verified: `if len(process_names) <= 1: continue`). Nothing fails
loudly at boot if these numbers are wrong — errors surface only as per-request admission rejects or
runtime OOM. The M3 smoke must include a peak-VRAM reading to validate the table empirically.

| component | size | notes |
|---|---|---|
| backbone weights (16L + text embed 128256×2048 + audio table 65632×2048) | ~2.74 GB | bf16; lm_head tied → free |
| cb0 head + depth decoder (4L + projector + heads `31×1024×2051`) | ~0.36 GB | bf16 |
| Mimi codec (shared encode+decode, `get_or_load_codec`) | ~0.4 GB | fp32 |
| KV pool @ max_running 4 | 256 MiB worst-case (4 × 64 MiB) | typical ≈ 27 MiB |
| KV pool @ max_running 8 | 512 MiB worst-case | typical ≈ 54 MiB |
| depth static KV + CG shadow buffers + staging | < 10 MiB | |
| dummy text logits (bs 8 × 128256 fp32) | 4 MiB/step | transient |
| CUDA graphs (M4, sizes {1,2,4,8}) | ~0.3–0.6 GB | capture pool |
| CUDA context + torch + fragmentation | ~1.2 GB | |
| **total @ mrr 8** | **≈ 5.6–6.0 GB** | **≥ 6 GB headroom on 12 GB** |

`mem_fraction_static = 0.5` (6 GB for weights+KV inside SGLang's accounting → ~2.9 GB KV pool ≈
**94k pooled positions**, 45× the worst-case need at mrr 8) — deliberately low to leave Mimi,
encoder activations, CG pool, and the colocated stages outside SGLang's fence. mrr 4 vs 8 changes
only the KV pool worst case (256 vs 512 MiB) and CG capture sizes; both fit trivially. The binding
constraint on the 3060 is **compute** (frame step must beat 80 ms RT), not VRAM.

---

## 4. Vocoder stage plan

Shape: `CsmStreamingVocoderScheduler(StreamingSimpleScheduler)` exactly per the higgs lifecycle
(latch contract, `_pending_done` buffering, abort → `clear_stream_state`, slim terminal result).

### 4.1 Frame math at 12.5 Hz (vs higgs 75 Hz defaults)
No delay pattern ⇒ **`raw_frames_available = len(rows)`** (higgs's `rows − N + 1` reversal warmup is
gone; frame 0 is decodable the moment it arrives). One row = one `[32]` tensor = 80 ms = 1920 samples.

| knob | higgs (75 Hz) | csm (12.5 Hz) | rationale |
|---|---|---|---|
| `stream_stride` (first decode) | 75 (≈1 s) | **13** (≈1.04 s) | same wall-clock warmth; overridden to 1 by the first-chunk knob on streaming paths |
| `stream_followup_stride` | 75 | **6** (480 ms) | steady chunk ≈ ½ s keeps Mimi calls chunky without bloating latency jitter |
| `stream_overlap_frames` | 8 (107 ms) | **4** (320 ms) | re-decode left context for the CNN receptive field; Mimi decode path has no conv-state cache (HF docstring: edge frames differ), 320 ms covers the upsampler ladder comfortably |
| `stream_holdback_frames` | 4 (53 ms) | **2** (160 ms) | never emit codec-edge frames mid-stream; final flush emits all |
| `initial_codec_chunk_frames` | param, raw-PCM default 1 | same — **default 1 frame** (80 ms) on raw-PCM/SSE streaming | the TTFA knob; clamped to steady size by `resolve_initial_codec_chunk_frames` |

Emission per delta: re-decode `[emitted − overlap : available − holdback]`, trim
`overlap × 1920` samples, emit exactly `new_frames × 1920` — seam-free without crossfade.
These values are starting points; M3 gate includes a seam-audibility A/B (overlap ∈ {2,4,6}).

### 4.2 The 2048 clamp + trim (three fences, in order)
1. Engine never streams EOS/STOP frames (§2.5).
2. Vocoder `sanitize_for_mimi`: count then `clamp_(0, 2047)` on every matrix entering Mimi —
   specials 2048/2049/2050 are OOB of the 2048-row Mimi codebooks (hard SIGSEGV-class hazard on
   some kernels, garbage on others). Non-zero clamp count is exported as a counter. **Scoping:**
   the `counter == 0` assertion holds only for **greedy/temp=0** runs (§6.2 parity gate); under
   `do_sample=True` the 2051-way heads can legally emit 2048–2050 with nonzero probability (HF
   itself has no clamp and would crash/corrupt) — in sampled mode (M5 benches) the counter is
   telemetry only, never a gate.
3. Final flush / non-streaming path: trim at the first **all-32-zero** frame (HF cutoff semantics,
   `generation_csm.py:471-477`) — mirrors HF's stop(31)/trim(32) inconsistency exactly.

### 4.3 Streaming state ownership
M2/M3: stateless overlap-trim decode (`codec.decode` per delta) — proven seam-free, zero
cross-request state by construction.
M4+: optional stateful path — per-request `CsmStreamState` owns `decoder_past_key_values`
(`codec.streaming_decode`), strictly per-request, dropped in `clear_stream_state`. Overlap re-decode
and stateful KV are mutually exclusive (re-decoding frames would double-advance the KV), so the
stateful path switches to holdback-only framing with a small fixed crossfade as fallback — gated on
an A/B against the stateless path before it becomes default. (The rime MimiCodecAdapter
state-leak postmortem is the cautionary tale here.)
Non-streaming requests: `_vocode_payloads` + `decode_batch` (exact-frame-count bucketed).

---

## 5. Config YAML + registry hookup

Registry: automatic. `import_pipeline_configs` pkgutil-scans `sglang_omni/models/*`, imports
`csm_tts` + `csm_tts.config`, finds `EntryClass`, registers `"CsmForConditionalGeneration"`.
`--model-path sesame/csm-1b` resolves via `config.json["architectures"][0]` — no `hf.py` map entry
needed. The ONE mandatory shared edit is the SGLang ModelRegistry dict (§1.16). Startup-debug rule:
grep the log for `"Ignore import error"` if the arch comes up unregistered.

`examples/csm_tts/csm-1b-rtx3060.yaml` — **`runtime_overrides[<stage>]` keys merge DIRECTLY as
factory kwargs** (no `factory_args:` nesting; `config/runtime.py:28-31,125-139` — a nested
`factory_args` key would arrive as a bogus kwarg and TypeError at stage build; flat shape confirmed
by `tests/unit_test/pipeline/test_runtime_adapter.py:50-56`). `server_args_overrides` dicts
deep-merge into the factory's own (`runtime.py:129-137`). `mem_fraction_static` may ride
`server_args_overrides` only while the typed `runtime.sglang_server_args.mem_fraction_static` is
unset (`runtime.py:82-91`) — we leave the typed field unset.
```yaml
config_cls: CsmTtsPipelineConfig
model_path: sesame/csm-1b
name: csm-1b-rtx3060
runtime_overrides:
  preprocessing:
    max_concurrency: 8
  audio_encoder:
    max_batch_size: 4
  tts_engine:
    max_new_tokens: 125
    enable_async_decode: false            # true from M4
    server_args_overrides:
      max_running_requests: 8
      mem_fraction_static: 0.5
      cuda_graph_max_bs: 8
      disable_cuda_graph: true            # false from M4
  vocoder:
    stream_followup_stride: 6
    stream_overlap_frames: 4
    stream_holdback_frames: 2
```
Launch: `python -m sglang_omni.cli serve --config examples/csm_tts/csm-1b-rtx3060.yaml`
(or `--model-path sesame/csm-1b` with library defaults). Dotted overrides
(`--stages.2.factory_args.server_args_overrides.max_running_requests 4`) work day one;
`--async-decode` works after the §1.16 optional CLI edit.

---

## 6. Test plan

### 6.1 Unit (CPU, no GPU, no sglang engine — higgs monkeypatch pattern)
- `test_modeling.py`: frame-embed compose vs hand-rolled `Σ embed[c_k + k*2051]` reference (random
  weights); `embed_codebook` offset correctness; `CsmCodebooksHead` index mapping (head k ↔ codebook
  k+1); **depth-decoder single-frame greedy parity vs HF `CsmDepthDecoderForCausalLM.generate`**
  (tiny-config random weights, temp=0, both fp32, exact code match) — this pins the
  position/offset/head off-by-ones (the #1 bug class here) before any GPU work.
- `test_sampler.py`: per-row `step_reference` vs `frame_finalize_direct` parity across phases
  (running / eos_now / done-freeze / mixed batch); EOS check uses cb0..30 only (cb31≠0 frame still
  stops — HF-exact); STOP sentinel freeze; greedy short-circuit determinism at temp=0/top_k=1;
  fixed-shape top-k filter vs naive per-row filter; **forced frame-0 EOS** (synthetic logits driving
  cb0..30 → 0 on the prefill path): request finishes at prefill, nothing emitted to the vocoder,
  finish reason set (F7 — the zero-decode lifecycle).
- `test_pipeline.py`: registry smoke (`PIPELINE_CONFIG_REGISTRY.get_config("CsmForConditionalGeneration")`);
  config topology (stream_to / terminal / can_accept_stream_before_payload / process="pipeline");
  factory defaults with monkeypatched `build_sglang_server_args` + `create_sglang_infrastructure`
  (asserts context_length 2048, disable_overlap_schedule, mrr 8); prompt builder vs the hub chat
  template token-for-token (golden token-id fixtures incl. 128000/128001/128002×F/128003);
  delay-free vocoder framing math (stride/overlap/holdback at 12.5 Hz, first-chunk knob, clamp
  counter, all-zero-frame trim, EOS frame never emitted); prefill-overlay paste positions incl.
  chunked-prefill cursor; runner finish-marking on synthetic model.
- `test_request_builders.py`: `normalize(None)` called; max_new_tokens clamp vs ctx budget;
  context fingerprint sensitivity (any code or speaker change → new key; no context → None).
- `test_async_decode_runner.py` (lands with M4): launch/resolve vs sync parity on mixed batches;
  next_token_ids published at launch; overrun-guard skips; bs=1 EOS edge; one real-pinned-memory test.

### 6.2 Golden parity vs HF (the correctness gate, M2)
`scripts/parity_csm_hf.py`: N=20 prompts (10 text-only, 10 with 3–10 s context audio), seeds fixed,
**temp=0 at BOTH levels** (cb0 greedy + depth greedy via `depth_decoder_do_sample=False`) so the run
is deterministic. Reference: `CsmForConditionalGeneration.generate(output_audio=False)` frame
sequences, run **in a subprocess** (isolates our `AutoConfig.register` override). Gate:
- engine in fp32: **exact integer frame-sequence match** (incl. the EOS frame), all 20 prompts;
- engine in bf16 (ship config): exact match through ≥ the first 25 frames AND ≥99% total code match,
  divergence only downstream of a first sampling tie-break;
- **Mimi-clamp guard: `sanitize_for_mimi` clamp counter == 0 across the whole run** (no code ≥2048
  ever reached the vocoder) — this assertion is valid ONLY because the run is greedy; under
  sampling the counter is telemetry, never a gate (§4.2);
- prompt-token parity: our `prompt_ids` == HF processor output for all 20;
- any **streamed**-waveform comparison vs HF excludes the final frame (§2.5 deviation: HF decodes a
  cb31≠0 EOS frame, we never stream it); non-streaming WAV comparisons are exact-trim-parity.
Note: HF's effective top_k=50 default arrives via `_get_default_generation_params()` at d557ef5
(GenerationConfig's attribute default is `None`) — treat any transformers bump that touches that
function as a parity-relevant event and re-run this gate.

### 6.3 Smoke on the 3060 (existing container: `docker start higgs`)
`docker exec higgs python -m sglang_omni.cli serve --config /workspace/csm-1b-rtx3060.yaml` then:
B=1 `POST /v1/audio/speech` non-streaming (WAV non-silent, X-Prompt/Completion-Token headers sane);
B=1 streaming raw-PCM (TTFA measured, monotone chunk cadence, no seam clicks by ASR/spectral check);
B=4 concurrent streams (per-request audio distinct, no cross-talk; temp=0 outputs identical to their
B=1 runs); abort mid-stream (KV released, vocoder state cleared, no "already freed" assert);
context-audio request (voice match audible; radix reuse on repeat request with same context);
**frame-0 EOS request** (forced via test hook or an empirically EOS-prone seed/prompt: server
returns a well-formed empty/near-empty audio response, KV released, vocoder stream closed cleanly).
Demo listening checks open in Chrome (MSE/chunked-WAV quirks).

---

## 7. Phased milestones + acceptance gates

| | scope | acceptance gate |
|---|---|---|
| **M1** skeleton + weights (~2 d) | files §1.1–1.8 + 1.16; loader handles the nonstandard `transformers.safetensors.index.json` shard naming; eager prefill smoke | registry resolves arch; all 538 tensors mapped or deliberately skipped (assert zero unexpected; the depth-embed tied duplicate counted as deliberately-skipped); fp32→bf16 load; single prefill forward on a dummy prompt returns finite hidden + a legal frame; unit tests for modeling/sampler green — **depth-decoder greedy parity test (pins the §1.4 head orientation) is a hard M1-exit blocker**; container checks: `AutoConfig.for_model("csm")` returns our subclass (§1.2) + `LlamaForCausalLM` 0.5.8 ctor/tie/final-norm parity (§1.7) |
| **M2** eager B=1 parity (~3 d) | runner §1.11 sync path, request_builders, preprocessing fast path, non-streaming vocoder | §6.2 gate in full (fp32 exact ×20, bf16 relaxed, clamp counter 0, prompt parity); non-streaming `/v1/audio/speech` returns intelligible audio (ASR WER sanity vs HF reference output) |
| **M3** CB B≤4 streaming via server (~3 d) | audio_encoder stage, streaming vocoder (stateless overlap-trim), stream metadata, abort path, YAML; `disable_cuda_graph=true` throughout | §6.3 smoke green incl. B=4 isolation + abort; temp=0 B=4 outputs bit-match their B=1 runs (paged backbone = no batch-composition drift — this is the explicit anti-left-pad-trap check); seam A/B picks overlap value; TTFA + RTF recorded as baseline |
| **M4** CUDA-graph depth + lookahead (~4 d) | capture full decode step (backbone + cb0 + unrolled 31-step depth) at bs {1,2,4,8}; enable async decode (min_bs 2); ping-pong pinned path; `test_async_decode_runner` | CG-on temp=0 parity vs M2 eager (exact); async-on overrun tests green (the three guards); perf: frame step p50 **< 80 ms at B=4** on the 3060 (RT at 4 streams), report B=8; no VRAM regression beyond the §3.3 CG line |
| **M5** head-to-head bench (~2 d) | bench vs the take-home stack (rime cuda + cuda_batched modes) + davidbrowne17/csm-streaming, same model, same 3060 | **frozen open-loop schedules** reused from the take-home harness (no turn-trimming; hit target context per press-test rule); one table with throughput AND latency together (TTFA p50/p99, TPOB, RTFx, max sustained streams); temp=0 cross-engine frame-match sanity on 3 prompts; writeup in the 3-pager register |

Pre-push gate at every milestone: full unit suite + fmt + clippy-equivalent (ruff) + the M-level
smoke; alpha-stage rules apply (pre-launch, wire formats and internal ABIs may break freely).

---

## 8. Risk register

| # | risk | detail | mitigation |
|---|---|---|---|
| R1 | OmniScheduler one-token-per-step assumptions vs the 31-step inner loop | Mostly aligned by construction (1 frame = 1 token = 1 KV position; `max_new_tokens` = frames, same as HF). Residual bite points: (a) **retraction** — `DecodeManager.retract` on KV pressure (`scheduling/sglang_backend/decode.py:35-80`) feeds `on_retract=lambda req: prefill_mgr.add_one_request(req)` (`scheduling/bootstrap.py:72-75`), replaying the prompt; the prefill path would sample a *new* frame 0 while `output_frames`/vocoder already consumed the old stream; sampler pool has no rollback (higgs has the same hole, documented at `models/higgs_tts/stages.py:67-68` — framed there against chunked prefill, same root fact). (b) decode step wall time is ~10–40 ms (variable with depth loop) — any upstream pacing heuristic tuned for ~1 ms LLM steps mismeasures (none exists today: `_event_loop_normal` is run-to-completion with a 1 ms idle sleep only, omni_scheduler.py:902-919). (c) logit processors run over our dummy zeros — harmless but must stay unsampled. | (a) make retraction structurally unreachable: admission check (input+frames ≤ max_req_len) + mrr 8 against a 45×-overprovisioned KV pool (§3.3); install the assert-on-retract in the **`on_retract` callable passed to bootstrap** — abort the request with an actionable error instead of silently re-prefilling. (b) keep `disable_overlap_schedule=True` (contract-required) and rely on omni's own loop; measure step-time histogram in M3. (c) `_finalize` already overwrites `output_ids` with cb0 (higgs pattern). |
| R2 | Weight-remap pitfalls | Nonstandard shard prefix `transformers-0000x-of-00002.safetensors` + `transformers.safetensors.index.json` may not match SGLang loader globs; `lm_head.weight` is the **cb0 head**, catastrophic if routed to a text head; depth embed is a tied alias that **is present** in the 538-tensor census (verified vs tf_index.json — the skip path always fires); orig-format `ckpt.pt`/`model.safetensors` in the same repo have **un-permuted RoPE** Q/K; ckpt is fp32. | M1 gate asserts exact tensor census (mapped/skipped/unexpected = 538/0 split); loader keys off the `transformers.safetensors.index.json` explicitly (fallback: one-time re-shard step in `resolve_checkpoint`); never read `ckpt.pt`; synthetic config `tie_word_embeddings=True` means no text lm_head exists to mis-load; unit test loads 2 layers' worth of fixture tensors through the mapper. |
| R3 | Processor/prompt formatting drift | Parity is hostage to token-exact prompt assembly: bos/eot framing per message, literal `[spk]` text, `<|AUDIO|>` expansion count F, `<|audio_eos|>` overlay, `add_special_tokens=False`. An F off-by-one shifts every pasted embedding. | Raw-waveform path computes F from the **actual Mimi encode** (placeholder assembly deferred to audio_encoder, §1.14) — the closed-form is used only for the pre-encoded fast path and is unit-tested against `MimiModel.get_encoded_length`; prompt-builder golden fixtures vs the hub chat template (§6.1); parity gate includes prompt-token equality (§6.2). |
| R4 | CUDA-graph capture of the depth loop | Anything data-dependent inside `forward` poisons capture; the 31-step loop bloats the graph (≈32 attn segments × 4 layers); multinomial-under-CG needs capturable RNG state. | M1–M3 run `disable_cuda_graph=true` — eager is the correctness baseline; M4 captures with the higgs discipline (only `_cg_*` slices in-graph, gather/scatter outside, fixed-shape topk, branchless greedy); parity gate re-run CG-on; if graph size/replay cost disappoints, fallback split: backbone+cb0 in-graph, depth loop eager in `post_decode` GPU code (contract-sanctioned alternative) — decision by measurement, not upfront. |
| R5 | Upstream velocity (higgs refreshed by #681/#692/#655 mid-flight — confirmed in git history: 0ab2e88 / 555fab5 / 2d4f758) | Base-runner hooks, OmniScheduler internals, StageConfig validation all move. | Pin `feat/csm-tts @ aabc487` for the build; mirror higgs file/hook structure 1:1 so rebases are mechanical diffs of a known shape; `test_pipeline.py` monkeypatch fixtures double as API-drift tripwires; rebase once at M3 and once before M5, never mid-milestone. |
| R6 | Mimi OOB / EOS frames reaching the codec | Codes 2048/2049/2050 index past the 2048-row codebooks; pad frames appear in finished rows. | Three fences (§4.2) + the clamp counter as a hard parity-gate assert; engine-side EOS frames are never streamed; non-streaming path trims at first all-32-zero frame. |
| R7 | `AutoConfig.register("csm", ..., exist_ok=True)` hijacks native CsmConfig in-process | The HF parity reference would silently pick up our subclass. | Wrapper subclasses `transformers.CsmConfig` (field-compatible, isinstance-true); parity harness runs HF in a subprocess regardless (§6.2); if transformers' CsmConfig gains a usable `get_text_config()` upstream, drop the registration and pass the synthetic config only inside `CsmTTSModel.__init__`. |
| R8 | Checkpoint-level bi-modal sampling (30–50% of `do_sample=True` runs cap out with degraded audio) | Known model property, not an engine bug — would pollute M5 quality comparisons and tempt bug-hunts. | All correctness gates at temp=0; sampled-mode benches use fixed seeds + report the clean/garbage split rate alongside (same protocol as the take-home stack so cross-engine comparison stays apples-to-apples); do not chase as a code bug. |
| R9 | bf16 numerics across the depth loop | 31 sequential bf16 matmul steps + fp32-trained ckpt; drift could shift sampling tie-breaks. | Logits fp32 before every sampler call (cb0 + 31 depth steps); fp32 engine mode retained as a debug flag; the structural B≥2 trap (dense left-pad batch) is absent by design — depth rows are all exactly 33 positions, right-aligned, no padding; the M3 B=4-vs-B=1 bit-match gate is the regression tripwire. |
| R10 | Colocated-stage GIL starvation | Audio encoder + vocoder + AR loop in one process (`process="pipeline"`); the documented 600× starvation. | Inherited for free (OmniScheduler 1 ms idle yield) — just don't remove it; M3 smoke includes concurrent encode-while-decoding. |

---

## Review resolutions

Two adversarial reviews folded in 2026-06-10: `/tmp/csm_sgl/review_framework.md` (framework-fit lens,
findings A1–A4 / B1–B30) and `/tmp/csm_sgl/review_model.md` (model-correctness lens, F1–F14).
Every REFUTED / UNCERTAIN / UNDERSPECIFIED finding and its disposition:

### REFUTED → fixed in plan
- **F4 (codebooks-head transpose)** — §1.4 formula was dimensionally invalid (`h @ weight[step].T`).
  Fixed to `h @ weight[step]` (≡ `F.linear(h, weight[step].T)`, HF modeling_csm.py:528-536), weight
  kept in checkpoint layout `[31,1024,2051]`, explicit warning against the re-store-transposed trap,
  and the §6.1 depth parity test promoted to a hard M1-exit blocker.
- **B11 (missing `next=` routing)** — §1.15 now requires `next=` on all three non-terminal stages;
  schema demands exactly one of `next`/`terminal` (`config/schema.py:298-302`); would have failed at
  startup as written.
- **B22 (YAML `runtime_overrides` nesting)** — §5 YAML flattened: per-stage keys merge directly as
  factory kwargs (`config/runtime.py:28-31,125-139`); the `factory_args:` nesting would have
  TypeError'd at stage build. Added `server_args_overrides` deep-merge + typed
  `mem_fraction_static` precedence notes. Re-verified directly in `runtime.py`/`schema.py` during
  this fold-in.
- **B6 (`seed` kwarg)** — §1.10 now uses `sampling_seed=int(seed)` (higgs request_builders.py:77-78).
- **A4/B18 (clamp off-by-one)** — `max_req_len = context_length − 1 = 2047`
  (omni_scheduler.py:150-153); §1.10 clamp corrected to `BACKBONE_CTX − 1 − len(prompt_ids)`; §3.1
  updated (incl. the cosmetic Qwen-flavored hint wording).

### UNDERSPECIFIED → specified
- **F7 (frame-0 EOS at prefill)** — the materially design-altering finding. Frame 0 now explicitly
  runs through `frame_finalize_direct` on the prefill path (§1.7, §2.4): on EOS → finished at
  prefill, nothing streamed, zero-decode lifecycle (KV release, `on_stream_done` with zero chunks,
  matching HF `cutoff_idx=0`). Empirically real for this checkpoint (rime #169). Forced frame-0-EOS
  unit test added to §6.1 and a smoke case to §6.3.

### UNCERTAIN → resolved by source check during this fold-in
- **B25 (LlamaForCausalLM composition)** — verified against upstream sglang `srt/models/llama.py`:
  ctor `(config, quant_config=None, prefix="")`; `tie_word_embeddings=True` → `lm_head` aliases
  `embed_tokens`, `load_weights` skips `lm_head.weight`; `rope_scaling` passed to `get_rope`.
  Checked at upstream HEAD (post-0.5.8); cheap 0.5.8-wheel re-confirm kept in the M1 gate.
- **B26 (final RMSNorm / post-norm hidden)** — verified: `LlamaModel.forward` applies
  `self.norm(hidden_states, residual)` on the last PP rank (llama.py:400). §1.7 updated with the
  receipt; combined with F2 (HF side CONFIRMED by the review), the post-norm conditioning chain is
  now verified end-to-end.
- **B19 (startup VRAM contract)** — verified `config/topology.py`: the colocation check skips when
  `len(process_names) <= 1`; all four stages share one process group → §3.3 re-labeled as unchecked
  engineering arithmetic with an empirical peak-VRAM reading added to the M3 smoke.
- **B29 (higgs churn PRs)** — confirmed in sglang-omni git history: 0ab2e88 (#681), 555fab5 (#692),
  2d4f758 (#655). R5 updated.
- **B3 (`AutoConfig.register exist_ok=True`)** — semantics documented in §1.2 (`_extra_content`
  consulted before the native mapping → in-process override wins; subclass keeps `model_type="csm"`
  so the consistency check passes); transformers is not importable on the dev box, so a one-liner
  container check stays in the M1 gate. R7 subprocess hedge unchanged.

### NOTE / caveat-level → documented
- **F8 (streamed EOS frame)** — §2.5/§6.2: streaming withholds the EOS frame that HF's non-streaming
  decode includes when cb31≠0 (≤80 ms shorter); final frame excluded from streamed-vs-HF waveform
  comparisons; non-streaming path is exact-parity.
- **F10 (top_k default mechanism)** — §6.2 note: effective top_k=50 comes from
  `_get_default_generation_params()` at d557ef5, not a GenerationConfig attribute default;
  transformers bumps touching it are parity-relevant.
- **F11 (clamp-counter scope)** — §4.2/§6.2: `counter == 0` is a gate only at temp=0; telemetry-only
  under sampling (HF itself has no clamp).
- **F13 (depth-embed duplicate)** — §1.6/R2: the tied `depth_decoder.model.embed_tokens.weight` IS
  in the 538-tensor census; skip path always fires; census counts it deliberately-skipped.
- **F14 (`encoded_audio_length` rationale)** — §1.3 corrected: the closed form serves pre-encode
  budget estimates for raw waveforms; the pre-encoded path uses `codes.shape[0]`; the
  defer-to-actual-encode rule is load-bearing and must not be optimized away.
- **A1 nits** — R1 citation corrected to `models/higgs_tts/stages.py:67-68` (chunked-prefill
  framing); assert-on-retract pinned to the `on_retract` callable (`scheduling/bootstrap.py:72-75`);
  step-pacing concern downgraded with the `_event_loop_normal` receipt.
- **B5 (Req signature)** — §1.10 now passes `origin_input_text=""` (positional signature).

### Confirmed solid (no change)
Depth conditioning chain (F1/F2/F3), frame-embed feedback (F5), EOS stop/trim semantics (F6),
KV growth/positions (F9), sampling defaults (F10 values), prompt formatting (F12), weight census
(F13 count), the in-forward multi-step decode trick (A1), optional-stage shape (A2), stream
routing/StreamItem contract (A3), admission-check mechanics (A4 per-request half), and B1–B30's
remaining confirmations (LOC calibration, hooks, adapters, registry scan, CLI overrides, GIL yield).
M4 depth-loop CUDA-graph capture remains the only unproven framework interaction — gated by design
(R4), unchanged.
