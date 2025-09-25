### Monarch-TorchFT-TorchTitan Distributed Training Orchestrator

#### Overview
This script orchestrates fault-tolerant distributed training using TorchTitan and TorchMonarch
frameworks. It manages multiple training replicas across SLURM-scheduled compute nodes
with automatic failure recovery and TorchFT lighthouse coordination.

##### PREREQUISITES
- Access to a SLURM cluster with GPU nodes
- TorchTitan training configuration file in script directory (debug_model.toml)
- A training dataset (c4_test) and tokenizer in script directory

##### CONFIGURATION
Before running, update the cluster-specific constants:
- MACHINE: TorchX named resource for your cluster (currently: "gpu.xlarge")
- MACHINE_MEMORY: Memory per machine in MB (currently: 2062607)
You can also override the resource configuration manually:
- https://docs.pytorch.org/torchx/main/specs.html#resource

##### USAGE
    python train_distributed.py --help

    Basic usage with 2 replicas, each with 1 node and 8 GPUs:
        python train_distributed.py

    Custom configuration:
        python train_distributed.py --replica-count 3 --gpu-per-node 8 \
            --host-per-replica 2 --training-steps 100

    With remote TorchFT lighthouse:
        python train_distributed.py --remote-lighthouse

##### KEY COMPONENTS
- LighthouseActor: Coordination server for fault tolerance
- TrainingActor: Individual trainer processes
- ReplicaActor: Manages groups of trainers
- OrchestrationManager: Top-level orchestration and failure recovery
- FailureController: Optional, periodically injects random failures into trainer processes

##### FAILURE RECOVERY
- Automatic retry with configurable delays (PER_ATTEMPT_DELAY)
- New allocations after repeated failures (PROC_ATTEMPTS)
- Maximum attempts per replica (MAX_ATTEMPT)

##### OUTPUT
- Training outputs saved to ./outputs directory
- Logs streamed from all distributed processes
- TensorBoard metrics enabled by default

##### CLEANUP
All SLURM jobs are automatically terminated at script completion.
