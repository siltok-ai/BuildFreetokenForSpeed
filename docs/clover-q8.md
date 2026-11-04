# Clover Q8 runtime and materials

The Clover profile appends one patch to the existing 65 patches and retains
the official upstream snapshot unchanged. It uses `build/profiles/cloverq8.json`,
version `0.1.3+speed.cloverq8.1`, four wheel native modules and 34 AOT DLL specs.
`integration/speed/current-release.json` selects the matching full rebuild. Pair
the checked-out revision with Speed through the release version, ABI, dependency,
native module and AOT inventory contracts. Record both repository commits; do not
combine this Speed release with the previous wheel/cache.

## Build

Use CUDA Toolkit 13.2, MSVC 19.44 and the locked runtime in build/README.md, with
`-Profile build/profiles/cloverq8.json` and a new output directory. Full compilation
is required: `_gguf` contains packed dense and expert CUDA operations. It is
loaded from the wheel. `FREETOKEN_DISABLE_JIT=1` prohibits native JIT; the explicit
development opt-in `FREETOKEN_GGUF_ALLOW_JIT=1` cannot override this prohibition.
Triton shape compilation remains enabled and its writable cache is separate.

## Model directory

Use a CPU environment with the locked tokenizer dependencies:

```powershell
python -B tools/clover_bundle.py --gguf D:\inputs\Clover-35B-A3B-Preview-Q8_0.gguf --output D:\models\clover-q8
```

The source lock fixes the official revision, length and SHA256. No conversion or
re-quantization is performed. The directory includes relative GGUF references,
tokenizer/template, generation defaults, manifest, attribution and `files.json`.
Copy the entire directory over an existing authenticated file transfer channel.
Verify each file against `files.json` after transfer. No serving endpoint is
created by the preparation tool.

The model uses `FreeTokenBackend`, resource profile `auto`, and defaults to causal
greedy generation. Requests can set `clover_mode` (`causal`, `selfspec`, `denoise`),
`clover_width` (8/16; denoise also 32), and `clover_threshold`. Block modes require
greedy sampling. Capacity is reserved for the declared maximum width at load.
Both 16GiB and 32GiB presets use 2304 expert slots, KV4096, prefill1024,
14GiB VRAM and 38GiB host budgets. Clover uses 512MiB VRAM and 512MiB Windows commit safety margins; Flash/Qwen retain the existing global margins. This fits
the 5070 Ti/64GiB Windows configuration; the 5090 preset awaits hardware tuning.
Larger expert/KV/prefill configurations require explicit GPU and host estimates.

Windows admission accounts for the 14GiB GPU mapping budget and 38GiB host estimate
together: 52GiB of commit, plus the 512MiB commit safety margin. The current system
snapshot therefore needs at least 52.5GiB of available commit. Fixed Q8 host weights
include 31.875GiB of routed expert banks and 0.9473GiB of BF16 input embeddings.
The host budget additionally covers the worker, loader temporaries and caches.
Two H3-to-Clover runs on the 5070 Ti, including all decode modes and inputs below
4K tokens, measured peak system commit increments of 50.21-51.51GiB and physical
RAM increments of 35.76-36.49GiB. The admission threshold exceeds the higher
observed commit increment by about 1GiB; it is not a guarantee under arbitrary
background workloads. Expert-cache capacity remains unchanged. Physical RAM capacity
alone does not establish that headroom: other processes, initialized model
frameworks and the configured paging-file capacity all count. The deployment
operator owns Windows paging-file sizing. Insufficient commit is rejected with
the required/current values and remains retryable after resources are available.

## Acceptance

CPU replay and contract checks do not qualify the binaries or model runtime.
Before promotion: compile the wheel and all specs, verify native inventory and
runtime DLL closure, then cold-start all three modes without MSVC/nvcc with native
JIT disabled. Check cancellation, invalid-request recovery, stop/output budgets,
prefix reuse, unload, and existing Flash Next MTP OFF/ON and Qwen models.
Record new execution results outside the repositories.
