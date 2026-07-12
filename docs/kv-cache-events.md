# Recording vLLM KV-cache events

SRT can enable vLLM's native KV-cache event publisher and subscribe to it
directly with Tachometer. This preserves fields that Dynamo's normalized event
path may filter or transform, which makes the trace suitable for diagnosing
cache churn and eviction-induced reuse misses.

```yaml
backend:
  type: vllm
  # Existing vllm_config remains unchanged.

telemetry:
  enabled: true
  container_image: telemetry-scraper
  dcgm_exporter:
    container_image: dcgm-exporter
    port: 9401
  node_exporter:
    container_image: node-exporter
    port: 9101
  kv_cache_events:
    enabled: true
    topic: ""
    jsonl_gz_roll_bytes: 268435456
    max_segments: 64
    ready_delay_ms: 500
    ready_timeout_secs: 600
```

When enabled, SRT:

1. Adds `--kv-events-config` to scheduler-bearing vLLM processes. Standard
   tensor-parallel endpoints publish from the leader; every explicit vLLM data
   parallel rank publishes separately.
2. Uses the existing topology allocator's collision-free KV-event ports and
   accounts for vLLM's data-parallel port offset.
3. Generates one `event_streams.sources` entry per publisher in
   `telemetry_config.toml`.
4. Starts Tachometer and waits for `telemetry/kv_events.ready` before starting
   the benchmark.
5. Stops workers before Tachometer so the recorder can drain and close its gzip
   segments before collection.

The collected artifacts are:

- `telemetry/kv-events/*.jsonl.gz`: raw decoded vLLM `EventBatch` records.
- `telemetry/kv_events_manifest.json`: per-source counts, sequence range and
  gaps, decode errors, trace paths, and overall completeness.
- `telemetry/kv_events.ready`: subscriber readiness marker used by SRT.

The first implementation detects and reports sequence gaps but does not request
replay from vLLM's optional replay socket. A complete manifest therefore means
the live subscriber observed a gap-free stream; an incomplete manifest should
not be used for exact eviction accounting without qualification.

Do not also set `kv-events-config` in `backend.vllm_config` when
`telemetry.kv_cache_events.enabled` is true. SRT rejects that ambiguous setup so
the advertised publisher endpoints cannot diverge from the Tachometer
subscriber topology.
