# Job port isolation

`job_scoped_ports: true` coordinates managed listeners for aggregate Dynamo/vLLM
jobs. Sidecars, connectors and shadow engines are not supported by this mode.
See [the example](../examples/vllm/job-port-isolation.yaml).

The Slurm job ID determines the first candidate among 15 slots. The controller
tries another slot if its worker or frontend reservations are occupied, or if
another job by the same user holds that slot on any participating node.
`SRTCTL_PORT_SLOT=1..15` pins the slot and disables this fallback.

The controller launches a small bare-host Python guard through srun on each
participating node, including dedicated infrastructure and benchmark nodes.
Guards hold node-local file locks until cleanup; they stop after workers,
frontends and services. The host must provide Python 3.10 or newer and the shared
checkout/output paths must be visible on every node. The guards use no container
and no host installation of srtctl.

## Requests and service retries

The existing topology allocator records worker reservations. Managed etcd, NATS
and exporter command builders request their ports when constructing commands.
Disabled, external and skipped services make no requests. Exporters with
different service names receive separate allocations. No global inventory of
arbitrary service commands or engine-internal sockets is maintained.

The selected slot provides 120 service ports (10128–11927 across all slots), separate from worker ranges.
Job-scoped service allocation replaces the managed exporters' `options.port`.
Their launch arguments, default readiness probes and default metrics annotations
read the selected allocation. Custom commands, arguments that override bind
flags, explicit readiness probes and explicit metrics annotations remain the
recipe's responsibility.

Before a new request is used, its node's guard probes it. Already accepted
requests are not probed again: a running service can keep its listener open
while the controller allocates another service. If a guard rejects a service
port, or an exited service logs a bind conflict during startup, the controller:

1. Stops every instance started for that service, leaving other services alive.
2. Checks that the instances stopped, then selects replacement service ports.
3. Preserves failed launch logs with `.port-attempt-N.out` suffixes.
4. Rebuilds and starts the whole service fleet with the replacement ports.

There are at most four attempts per service. Unrelated errors are not retried.
Workers and dependent stages start only after the service's readiness gate
passes. A failed guard step aborts startup; missing Python, permission errors
and guard timeouts do not trigger a search for another slot. Services that fail
after startup use the existing process-monitor failure path, not port retries.

## Evidence and limits

`logs/port_plan.json` contains the selected slot, job ID, node set, revision,
fixed service ports, worker ranges and individual requests. Each request names
its owner, node, transport, bind address and span. A worker reservation without
a node applies conservatively to all worker nodes. `port_allocation_attempts.jsonl`
records rejected guard checks and service bind retries. Failed service logs preserve launch errors.

Leases coordinate cooperating launchers by the same user. Other users and
unmanaged applications do not participate in these locks. Probes close their
sockets before service launch; they are availability checks, not kernel socket
reservations. No unrelated listener is killed. Legacy launchers with the old
slot layout must not share nodes with this layout; use a separate allocation
while introducing it.

vLLM's scan bases are recorded as `enforced: false`: they do not enforce an upper
bound and are not advertised as protected listener ranges. Uncontrolled
engine-internal sockets, arbitrary custom services, and the separate power
telemetry launch path are outside this mode's automatic service retry support.
The worker ranges remain bounded per kind; exhaustion fails rather than
allocating into another slot. Host ephemeral-port policy is unchanged.

## Local validation

```bash
uv sync --group dev
uv run pytest tests/test_port_reservation.py tests/test_job_ports.py
uv run srtctl dry-run -f examples/vllm/job-port-isolation.yaml
```

The fault tests use real local sockets and guard subprocesses, with srun replaced
by local subprocess transport. They cover occupied ports, conflicts on later
nodes, lease release, slot fallback, and whole-fleet service retries. They do not
launch a model or qualify Slurm transport, GPU serving, or a live campaign.
