# LocalSGD Example

This example demonstrates localSGD training with torchft.

From the docstrings of the LocalSGD class:

```txt
LocalSGD is a context manager that
implements the algorithm described in https://arxiv.org/pdf/1805.09767

This will synchronize the model parameters periodically in a fault tolerant
way using a torchft Manager. The allreduce on the parameters will happen
every sync_every steps after the optimizer.step call.

The torchft quorum is computed at the beginning of ``sync_every`` steps. If
any error occurs, or a worker fails between syncs, ``sync_every`` steps will be
discarded and a new quorum will be computed on the next step.

If running in async mode, on a joining worker the first ``sync_every`` steps
will discarded as the model will be recovering during that period. When
using sync mode, the checkpoint will be restored prior to the first step.
```

## Implementation Details

For localSGD training, there is no need to wrap the optimizer with the [OptimizerWrapper](../../torchft/optim.py#L24). This is because the LocalSGD context manager handles the calls to `manager.start_quorum()` and `manager.should_commit()`

## How to Run

These assumes that you are in the root directory of the torchft repository.

1. Start the Lighthouse server:
```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

2. Run using torchx:
```bash
cd examples/localsgd
torchx run
```

3. Or manually run multiple replica groups:

Shell 1 (first replica group):
```bash
export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=2
CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29600 --nnodes=1 --nproc_per_node=1 examples/localsgd/train_localsgd.py
```

Shell 2 (second replica group):
```bash
export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=2
CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29601 --nnodes=1 --nproc_per_node=1 examples/localsgd/train_localsgd.py
```

## Interprating Outputs

You should see snippets like the following from the output:

```sh
LocalSGD: Number of local optimizer steps completed: 100
```

And 

```sh
2025-05-22T10:52:30.575 [INFO] [torchft::manager] - [Replica train_ddp_0] got lighthouse quorum LighthouseQuorumResponse { quorum: Some(Quorum { quorum_id: 3, participants: [QuorumMember { replica_id: "train_ddp_0:4e6f882a-35c9-4f50-af99-ee1e0aae1310", address: "http://sz-k8s-master:45321", store_address: "127.0.0.1:29600", step: 8, world_size: 1, shrink_only: false, commit_failures: 0, data: "" }], created: Some(Timestamp { seconds: 1747882350, nanos: 574746342 }) }) }
```

The step above is the number of local optimizer step completed by localSGD, and the step below in QuorumMember is the number of global sync steps completed.