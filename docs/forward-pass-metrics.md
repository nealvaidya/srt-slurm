# Dynamo forward-pass metrics

SRT can collect Dynamo `ForwardPassMetrics` into the same Tachometer dataset as
DCGM, node, backend, and frontend metrics. This path is intended for static
SLURM deployments: expected publishers are derived from the allocated endpoint
topology, and the benchmark waits until Tachometer has stored at least one FPM
event from every logical worker.

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
    namespace: dynamo
    ready_timeout_secs: 600
    connect_timeout_secs: 120
```

SRT then:

- sets `DYN_EVENT_PLANE=zmq` and `DYN_FORWARDPASS_METRIC_PORT` on every backend
  worker, with collision-free base ports for co-located workers;
- leaves `DYN_REQUEST_PLANE=nats` unchanged for the serving deployment;
- starts the Dynamo FPM exporter on the head node with an etcd discovery client
  and TCP request-plane runtime, so the exporter itself does not require NATS;
- connects the exporter to Tachometer over a head-node-local Unix socket; and
- waits for `telemetry/fpm.ready` before starting the benchmark.

The resulting `telemetry/final.parquet` contains `dynamo_fpm_*` scalar metrics.
Useful join and integrity columns include `worker_role`, `worker_id`, `dp_rank`,
`fpm_counter_id`, `fpm_publisher_id`, and `fpm_event_sequence`. Shutdown also
writes `telemetry/fpm_manifest.json` with publisher coverage and detected event
or counter gaps.

The selected Dynamo installation must contain
`dynamo.common.export_forward_pass_metrics` and the envelope-aware
`FpmEventSubscriber.recv_envelope()` API. The Tachometer image must contain the
matching FPM listener support.
