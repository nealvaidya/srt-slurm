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
  also include the job ID. Slots 1 through 15 are deterministic. A private,
  node-local user lock holds the slot until orchestration and process cleanup
  finish. Jobs congruent modulo 15 fail before launching services when that
  slot is already held; choose distinct `SRTCTL_PORT_SLOT` values or resubmit.
  Kernel process exit releases the lock, including after crashes. The lease
  does not change the locked recipe's port assignment. Occupied infrastructure
  ports also fail before discovery-plane startup. Unsupported deployments fail
  validation. Other users and older jobs do not participate in this user lock;
  it cannot guarantee isolation against their concurrent use of arbitrary ports.
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
