# CLU compatibility patches on schema 2

This branch starts at upstream 898e695b5b5c358e90915c361ad7c97d28e7edde
(v2.42.3). It carries the small compatibility contract required by CLU.

- Job-owned `reporting.s3` wins over mutable cluster configuration. Raw artifacts
  upload as individually verifiable keys when the recipe sets `exclude: []` and
  `archive: []`. Credential values are redacted from command logging.
- Trace-replay warmup adds a default 512-token cap after ordinary extra inputs.
  The measured replay preserves workload parameters.
- `job_scoped_ports: true` assigns bounded port slots for the supported
  single-node aggregate Dynamo/vLLM deployment. Infrastructure state directories
  also include the job ID. Slots 1 through 15 are deterministic, not a lease;
  job IDs congruent modulo 15 can collide. Choose distinct `SRTCTL_PORT_SLOT`
  values for such concurrent jobs. Unsupported deployments fail validation.
  This does not guarantee isolation from older jobs using arbitrary ports.
- `services[].type: kv-events` runs a required head-node recorder independently
  from native metrics and FPM. Every worker role must publish KV events. Source
  endpoints follow allocated listener blocks, including local DP ranks and
  vLLM's global-rank endpoint offset. Sidecars, shadow engines, and cross-node
  DP replicas are outside this recorder contract.

The KV recorder uses the existing pinned container binary. With zero metrics
endpoints its main task set would complete immediately; `--sync-interval 30`
keeps its normal periodic sync task running until graceful termination. It
starts after workers/frontend and waits for subscriber readiness. Workers stop
before the recorder drains. Finalization requires complete, positive, gapless
capture from the allocated fleet and all declared nonempty trace files.

FPM has not been ported. Native metrics continue to use upstream Tachometer;
required KV capture uses its separate service and manifest. These patches do
not disable or replace upstream power, profiling, or dashboard features.
