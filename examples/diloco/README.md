# DiLoCo Example

This example demonstrates DiLoCo training.

From the doc strings of the [DiLoCo class](../../torchft/local_sgd.py#L157):

```txt
DiLoCo is a subclass of LocalSGD that overrides the synchronization
mechanism to average and synchronize the pseudogradients (delta of the previous global weight and current local weights).

This algorithm requires a backup copy of the
weights. By default these are stored in CPU memory. If any error occurs
during the DiLoCo step, the step will be discarded and the model
parameters will reset back to the last time DiLoCo synchronized.

DiLoCo paper: https://arxiv.org/pdf/2311.08105
```

## Implementation Details

As seen in the training script, DiLoCo defines two optimizers, one for the inner loop and one for the outer loop. The paper found that using Adam for the inner loop and SGD for the outer loop worked best over a range of configurations.

A backup device is specified. This backup device is used to store a copy of the model parameters at the beginning of the inner optimization loop, so that the outer step can be applied to the model parameters before undergoing inner optimization, keeping the model parameters in sync.

## How to Run

These assumes that you are in the root directory of the torchft repository.

1. Start the Lighthouse server:
```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

2. Run using torchx:
```bash
cd examples/diloco
torchx run
```

3. Or manually run multiple replica groups:

Shell 1 (first replica group):
```bash
export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=2
CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29600 --nnodes=1 --nproc_per_node=1 examples/diloco/train_diloco.py
```

Shell 2 (second replica group):
```bash
export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=2
CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29601 --nnodes=1 --nproc_per_node=1 examples/diloco/train_diloco.py
```

## Example Outputs

You should see snippets like the following from the output:

```sh
DiLoCo: Number of inner optimizer steps completed: 500
DiLoCo: Number of outer optimizer steps completed: [5] loss = 7.605193614959717
```

These tell you that the inner optimizer has completed 500 steps and the outer optimizer has completed 5 steps.


