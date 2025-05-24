# TorchFT Examples

This directory contains advanced examples demonstrating various fault tolerance features and training approaches in TorchFT beyond the basic `train_ddp.py` example in the [README](../README.md).

Each directory contains a README with more detailed instructions, as well as extensive documentation on the feature being showcased and how to interpret the outputs.

## List of Examples

- [DDP with proactive failure recovery](./ddp_proactive/README.md): Demonstrates DDP with proactive failure recovery mode
- [DiLoCo](./diloco/README.md): Demonstrates Distributed Local Convergence training
- [LocalSGD](./localsgd/README.md): Demonstrates Local SGD with periodic synchronization
- [Live Checkpoint Recovery](./live_checkpoint_recovery/README.md): Demonstrates live checkpoint recovery

## Running the examples

After starting the lighthouse server by running:

```sh
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

You can `cd` into the example directory:

```sh
cd examples/[example_directory]
```

and then launch the example with torchX with:

```sh
export QUICK_RUN=1
torchx run
```

the QUICK_RUN environment variable runs the examples for much less steps, and also uses a synthetic, rather than downloaded, dataset. It is useful for testing the examples quickly.

See the `.torchxconfig` file in each example directory for configuration details, and [torchx.py](../torchft/torchx.py) and the [torchX documentation](https://pytorch.org/torchx/latest/) to understand how DDP is being ran. 