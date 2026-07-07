# Dynamo forward-pass metrics

SRT can collect Dynamo `ForwardPassMetrics` into the same Tachometer dataset as
DCGM, node, backend, and frontend metrics. This path is intended for static
SLURM deployments. Dynamo workers write bounded, rotating gzip JSONL traces to
the shared run directory, and Tachometer imports them into `final.parquet`
during graceful shutdown.

Enable it under the existing telemetry configuration:

```yaml
telemetry:
  enabled: true
  container_image: <tachometer-image-with-fpm-support>
  dcgm_exporter:
    container_image: <dcgm-exporter-image>
    port: 9401
  node_exporter:
    container_image: <node-exporter-image>
    port: 9101
  forward_pass_metrics:
    enabled: true
    mode: full
    max_segments: 64
    ready_timeout_secs: 600
```

SRT then:

- sets `DYN_EVENT_PLANE=zmq`, `DYN_FORWARDPASS_METRIC_PORT`, and Dynamo's
  `DYN_FPM_*` trace settings on every backend worker, with collision-free base
  ports for co-located workers;
- leaves `DYN_REQUEST_PLANE=nats` unchanged for the serving deployment;
- writes producer-specific segments under
  `telemetry/fpm/dynamo-fpm.<producer>.<segment>.jsonl.gz`;
- waits for every expected producer to open its first trace segment before
  starting the benchmark; and
- imports all completed segments into Tachometer after workers stop and before
  Tachometer compacts its dataset.

The resulting `telemetry/final.parquet` contains `dynamo_fpm_*` scalar metrics.
Useful join and integrity columns include `worker_role`, `worker_id`, `dp_rank`,
`fpm_counter_id`, `fpm_producer_id`, and `fpm_capture_mode`. Shutdown also
writes `telemetry/fpm_manifest.json` with source files, worker and producer
coverage, invalid record counts, and detected FPM counter gaps.

The selected Dynamo installation must include the producer-side FPM tracing
added by Dynamo PR #11110. The Tachometer image must contain the matching
`dynamo.fpm.trace.v1` importer.

`mode` defaults to `full` because benchmark analysis requires every forward
pass. Dynamo itself defaults to five-second sampling, so do not omit this SRT
setting when full iteration history is required. `max_segments` defaults to 64
instead of Dynamo's four-segment default to avoid pruning data during multi-hour
runs. Increase it further if the configured roll size and expected forward-pass
rate require more retention.
