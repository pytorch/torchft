# Live Checkpoint Recovery Example

This example demonstrates live checkpoint recovery in torchft using [process group based transport](../../torchft/checkpointing/pg_transport.py).

The description of Live Checkpoint Recovery from [Fault Tolerance Poster](../../media/fault_tolerance_poster.pdf)

```txt
# Live Checkpoint Recovery
- We’re developing a novel way to live recover from failures by asynchronously saving checkpoints and serving them directly to newly joined and recovering workers.
- On worker start, the checkpoint is transferred via HTTP from an existing healthy worker.
- The weights are copied from the GPU in a non-blocking way during the forward pass using a separate CUDA stream.
- We use leader election to identify live workers and exchange step information to recover from failures.
```

## Implementation Details

The example is based on [train_ddp.py](../../train_ddp.py). We add the following logic before the construction of the `Manager` object:

```python
if REPLICA_GROUP_ID == 0:
    time.sleep(10)
```

This simulates a worker that joins the training in the middle. Because the worker that joins has a step value that is less than the step value of the workers currently in training, Live Checkpoint Recovery will be triggered.


## How to Run

You can experiment with live checkpoint recovery mode by launching the following commands in two shells at around the same time.

On shell 1 (one replica group starts initial training):
```sh
export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=2
export TORCHFT_PROACTIVE_RECOVERY=1

CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29600 --nnodes=1 --nproc_per_node=1 -- examples/live_checkpoint_recovery/train_ddp_lcr.py
```

On shell 2 (a second replica group joins):
```sh
export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=2
export TORCHFT_PROACTIVE_RECOVERY=1

CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29601 --nnodes=1 --nproc_per_node=1 -- examples/live_checkpoint_recovery/train_ddp_lcr.py
```

## Example Outputs

Below are snippets that you should see from the terminal outputs. Comments have been added to faciliate understanding.

You should see snippets like the following from the output for Replica Group ID 0 that shows the steps of Live Checkpoint Recovery.

```txt
2025-05-22T11:16:34.425 [INFO] [torchft::manager] - Creating LighthouseClient: establishing connection to http://localhost:29510

2025-05-22T11:16:34.817 [INFO] [torchft::manager] - [Replica train_ddp_0] Start quorum for group_rank 0

2025-05-22T11:16:34.817 [INFO] [torchft::manager] - [Replica train_ddp_0] All workers joined - starting quorum

2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_0] got lighthouse quorum LighthouseQuorumResponse { quorum: Some(Quorum { quorum_id: 12, participants: [QuorumMember { replica_id: "train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471", address: "http://sz-k8s-master:33997", store_address: "127.0.0.1:29600", step: 0, world_size: 1, shrink_only: false, commit_failures: 0, data: "" }, QuorumMember { replica_id: "train_ddp_1:99b3452a-ef27-4523-8b4f-cdab6cbbc004", address: "http://sz-k8s-master:42813", store_address: "127.0.0.1:29601", step: 17, world_size: 1, shrink_only: false, commit_failures: 0, data: "" }], created: Some(Timestamp { seconds: 1747883794, nanos: 905534965 }) }) } # train_ddp_0 has step=0, as it just joined, whilst train_ddp_1 has step=17

2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_0] Finished quorum for group_rank 0

2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_0] healing is required step=0, max_step=17, recover_src_replica_rank=1 # This discrepancy in the steps triggers live checkpoint recovery. train_ddp_0 initiates an MPI recv call to recover_src_replica_rank

INFO:torchft.manager:[train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471/0 - step 0] reconfiguring for quorum_id=12 store_prefixed_addr='127.0.0.1:29601/torchft/12/0' # Process group needs to be reconfigured after a new replica joins. The rendezvous store adds a new prefixed address to prevent data from previous rendezvous attempts affecting the current rendezvous

INFO:torchft.manager:[train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471/0 - step 0] healing required, fetching checkpoint metadata from recover_src_manager_address='http://sz-k8s-master:42813' max_step=17

2025-05-22T11:16:34.910 [INFO] [torchft::manager] - Creating ManagerClient: establishing connection to http://sz-k8s-master:42813
INFO:torchft.manager:[train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471/0 - step 0] fetching checkpoint from recover_src_replica_rank=1 with checkpoint_metadata='<n/a>' # Checkpoint metadata is only needed if using a checkpoint server. Here, we directly use MPI send/recv

[W522 11:16:35.452918836 reducer.cpp:1430] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())

INFO:torchft.manager:[train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471/0 - step 17] applying pending state dict # State dict contains the optimizer state and model states that the recover_src_replica_rank sends over.

INFO:torchft.manager:[train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471/0 - step 17] Loaded state dict.
```

And from the output for Replica Group ID 1 that shows the steps of Live Checkpoint Recovery.

```txt
2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_1] Start quorum for group_rank 0

2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_1] All workers joined - starting quorum

2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_1] got lighthouse quorum LighthouseQuorumResponse { quorum: Some(Quorum { quorum_id: 12, participants: [QuorumMember { replica_id: "train_ddp_0:9f5b5624-112d-4b51-a995-c8a08b640471", address: "http://sz-k8s-master:33997", store_address: "127.0.0.1:29600", step: 0, world_size: 1, shrink_only: false, commit_failures: 0, data: "" }, QuorumMember { replica_id: "train_ddp_1:99b3452a-ef27-4523-8b4f-cdab6cbbc004", address: "http://sz-k8s-master:42813", store_address: "127.0.0.1:29601", step: 17, world_size: 1, shrink_only: false, commit_failures: 0, data: "" }], created: Some(Timestamp { seconds: 1747883794, nanos: 905534965 }) }) } # train_ddp_0 has step=0, as it just joined, whilst train_ddp_1 has step=17, this should be the same as the output for REPLICA Group ID 0 

2025-05-22T11:16:34.905 [INFO] [torchft::manager] - [Replica train_ddp_1] Finished quorum for group_rank 0 

INFO:torchft.manager:[train_ddp_1:99b3452a-ef27-4523-8b4f-cdab6cbbc004/0 - step 17] reconfiguring for quorum_id=12 store_prefixed_addr='127.0.0.1:29601/torchft/12/0' # Process group needs to be reconfigured after a new replica joins. The rendezvous store adds a new prefixed address to prevent data from previous rendezvous attempts affecting the current rendezvous

INFO:torchft.manager:[train_ddp_1:99b3452a-ef27-4523-8b4f-cdab6cbbc004/0 - step 17] peers need recovery from us [0]
INFO:torchft.checkpointing.pg_transport:preparing state_dict took 0.0023363223299384117s # [0] is the replica_group_rank of the peer that needs recovery. Does an MPI send to that replica_group_rank

/srv/apps/torchft/torchft/checkpointing/pg_transport.py:208: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1577.)
  buf_t = torch.frombuffer(buf, dtype=torch.uint8).to(self._device)
INFO:torchft.checkpointing.pg_transport:send pickle took 0.08277195505797863s # Time taken to pickle the tensor for sending
INFO:torchft.checkpointing.pg_transport:send tensors took 1.706640336662531s # Time taken until the tensor is sent
```