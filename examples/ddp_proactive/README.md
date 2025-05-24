# DDP Proactive Recovery Example

This example demonstrates DDP with proactive failure recovery in torchft.

Proactive recovery enables the training process to quickly detect and respond to worker failures without waiting for timeout periods, significantly reducing recovery time.

Note that the setting `TORCHFT_PROACTIVE_RECOVERY=0` does not disable the Lighthouse heartbeat timeout detection logic, but stops the processes from spawning up a listening process.

## Implementation Details

The example is based on [train_ddp.py](../../train_ddp.py). We add the following logic before the construction of the `Manager` object:

```python
            if manager.current_step() == 3:
                if REPLICA_GROUP_ID == 0:
                    manager.shutdown()
                    exit(0)
                # If proactive recovery, then the surviving process will reconfigure
                # If not proactive recovery, then the surviving process will wait until timeout
            test_tensor = torch.tensor([1.0]).to(device)
            manager.allreduce(test_tensor)
```

Here, without proactive error recovery, after Replica Group ID 0 shuts down, Replica Group ID 1 will wait until all reduce timeout (set to 120 seconds by default).

However, with proactive error recovery enabled, the Lighthouse will detect that Replica Group ID 0 heartbeat times out and sends a message to Replica Group ID 1 to reconfigure its process group to exclude the failed replica.

## How to Run

You can experiment with proactive failure recovery mode by:

```sh
export TORCHFT_PROACTIVE_RECOVERY=1
```

On shell 1 (one replica group starts initial training):
```sh
export REPLICA_GROUP_ID=0
export NUM_REPLICA_GROUPS=2
export TORCHFT_PROACTIVE_RECOVERY=1

CUDA_VISIBLE_DEVICES=0 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29600 --nnodes=1 --nproc_per_node=1 -- examples/ddp_proactive/train_ddp_proactive.py
```

On shell 2 (a second replica group joins):
```sh
export REPLICA_GROUP_ID=1
export NUM_REPLICA_GROUPS=2
export TORCHFT_PROACTIVE_RECOVERY=1

CUDA_VISIBLE_DEVICES=1 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port=29601 --nnodes=1 --nproc_per_node=1 -- examples/ddp_proactive/train_ddp_proactive.py
```

And contrast this with if you did:

```sh
export TORCHFT_PROACTIVE_RECOVERY=1
```

## Example Outputs

### With `TORCHFT_PROACTIVE_RECOVERY=1`

#### Lighthouse

You should see the following output from the Lighthouse:

```txt
2025-05-22T11:47:49.435 [INFO] [torchft::lighthouse] - Replica train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6 timed out (last heartbeat: Instant { tv_sec: 5334992, tv_nsec: 45173926 }), sending failure notification.
2025-05-22T11:47:49.435 [INFO] [torchft::lighthouse] - Removed replica train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6 from heartbeats and participants due to timeout.
2025-05-22T11:47:49.435 [INFO] [torchft::lighthouse] - New failure detected, resetting all participants for quorum formation.
2025-05-22T11:47:49.435 [INFO] [torchft::lighthouse] - Healthy replicas received failure notification for train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6 with error message: heartbeat timeout
```

Here, the Lighthouse detect heartbeat timeout and sends failure notifications to the healthy replicas.

#### Replica Group ID 0 

You should see the following output from Replica Group ID 0:

```txt
INFO:torchft.manager:[train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6/0 - step 3] Setting error processor thread stop event
INFO:torchft.manager:[train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6/0 - step 3] Waiting for error processor thread to complete
INFO:torchft.manager:[train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6/0 - step 3] Error processor thread shutdown completed.
INFO:torchft.manager:[train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6/0 - step 3] Setting failure listener stop event for process
INFO:torchft.manager:[train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6/0 - step 3] Waiting for failure listener process to complete
INFO:torchft.manager:[train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6/0 - step 3] Failure listener process shutdown completed
```

This is the shutdown logic of the error processor thread and failure listener process. The failure listener process listens to the failure notifications from the Lighthouse, and transmits it to the error processor thread in the main training process.

#### Replica Group ID 1

Replica Group ID 1 will recovery quickly after the shutdown of Replica Group ID 0. In the middle, there will be errors relating to the TCPStore due to Replica Group ID 1 aborting its process group in the middle of allreduce. The output is shown in full to show that these error traces are expected.

```txt
[W522 11:47:45.949247853 TCPStore.cpp:125] [c10d] recvValue failed on SocketImpl(fd=77, addr=[::ffff:127.0.0.1]:39732, remote=[::ffff:127.0.0.1]:29600): failed to recv, got 0 bytes
Exception raised from recvBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:678 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f9a28f795e8 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x5ba8afe (0x7f9a6d09cafe in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x5baae40 (0x7f9a6d09ee40 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x5bab74a (0x7f9a6d09f74a in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x2a9 (0x7f9a6d0991a9 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7f9a2a2929a9 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xdbbf4 (0x7f9a1a25bbf4 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/bin/../lib/libstdc++.so.6)
frame #7: <unknown function> + 0x8609 (0x7f9a84eb4609 in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #8: clone + 0x43 (0x7f9a84c7f353 in /lib/x86_64-linux-gnu/libc.so.6)

[W522 11:47:45.952720714 ProcessGroupNCCL.cpp:1659] [PG ID 0 PG GUID  Rank 1] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: failed to recv, got 0 bytes
[W522 11:47:46.952883182 TCPStore.cpp:106] [c10d] sendBytes failed on SocketImpl(fd=77, addr=[::ffff:127.0.0.1]:39732, remote=[::ffff:127.0.0.1]:29600): Broken pipe
Exception raised from sendBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:653 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f9a28f795e8 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x5ba8afe (0x7f9a6d09cafe in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x5baa358 (0x7f9a6d09e358 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x5babb3e (0x7f9a6d09fb3e in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x298 (0x7f9a6d099198 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7f9a2a2929a9 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xdbbf4 (0x7f9a1a25bbf4 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/bin/../lib/libstdc++.so.6)
frame #7: <unknown function> + 0x8609 (0x7f9a84eb4609 in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #8: clone + 0x43 (0x7f9a84c7f353 in /lib/x86_64-linux-gnu/libc.so.6)

[W522 11:47:46.955975324 ProcessGroupNCCL.cpp:1659] [PG ID 0 PG GUID  Rank 1] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Broken pipe
[W522 11:47:47.956137177 TCPStore.cpp:106] [c10d] sendBytes failed on SocketImpl(fd=77, addr=[::ffff:127.0.0.1]:39732, remote=[::ffff:127.0.0.1]:29600): Broken pipe
Exception raised from sendBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:653 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f9a28f795e8 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x5ba8afe (0x7f9a6d09cafe in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x5baa358 (0x7f9a6d09e358 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x5babb3e (0x7f9a6d09fb3e in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x298 (0x7f9a6d099198 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7f9a2a2929a9 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xdbbf4 (0x7f9a1a25bbf4 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/bin/../lib/libstdc++.so.6)
frame #7: <unknown function> + 0x8609 (0x7f9a84eb4609 in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #8: clone + 0x43 (0x7f9a84c7f353 in /lib/x86_64-linux-gnu/libc.so.6)

[W522 11:47:47.959256423 ProcessGroupNCCL.cpp:1659] [PG ID 0 PG GUID  Rank 1] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Broken pipe
[W522 11:47:48.959394571 TCPStore.cpp:106] [c10d] sendBytes failed on SocketImpl(fd=77, addr=[::ffff:127.0.0.1]:39732, remote=[::ffff:127.0.0.1]:29600): Broken pipe
Exception raised from sendBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:653 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f9a28f795e8 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x5ba8afe (0x7f9a6d09cafe in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x5baa358 (0x7f9a6d09e358 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x5babb3e (0x7f9a6d09fb3e in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x298 (0x7f9a6d099198 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7f9a2a2929a9 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xdbbf4 (0x7f9a1a25bbf4 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/bin/../lib/libstdc++.so.6)
frame #7: <unknown function> + 0x8609 (0x7f9a84eb4609 in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #8: clone + 0x43 (0x7f9a84c7f353 in /lib/x86_64-linux-gnu/libc.so.6)

[W522 11:47:48.962514781 ProcessGroupNCCL.cpp:1659] [PG ID 0 PG GUID  Rank 1] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Broken pipe
INFO:torchft.manager:[train_ddp_1:9b6cf09c-8747-43dd-bf2c-067cc4d77550/0 - step 3] Received error: Peer failure detected in listener process: replica train_ddp_0:782b3df7-ac82-4d1c-9c95-ff05b4c2ddb6 has failed
NoneType: None

NoneType: None
```

### With `TORCHFT_PROACTIVE_RECOVERY=0`

Execute the following command on Replica Group ID 1:
```sh
export TORCHFT_PROACTIVE_RECOVERY=0
```

You should observe that Replica Group ID 1 stalls for 30 seconds before resuming training.

```txt
[W522 11:47:47.959256423 ProcessGroupNCCL.cpp:1659] [PG ID 0 PG GUID  Rank 1] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Broken pipe
[W522 11:47:48.959394571 TCPStore.cpp:106] [c10d] sendBytes failed on SocketImpl(fd=77, addr=[::ffff:127.0.0.1]:39732, remote=[::ffff:127.0.0.1]:29600): Broken pipe
Exception raised from sendBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:653 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7f9a28f795e8 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x5ba8afe (0x7f9a6d09cafe in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x5baa358 (0x7f9a6d09e358 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x5babb3e (0x7f9a6d09fb3e in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: c10d::TCPStore::check(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&) + 0x298 (0x7f9a6d099198 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: c10d::ProcessGroupNCCL::heartbeatMonitor() + 0x379 (0x7f9a2a2929a9 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #6: <unknown function> + 0xdbbf4 (0x7f9a1a25bbf4 in /srv/apps/danny/miniconda3/envs/warren/torchtitan/bin/../lib/libstdc++.so.6)
frame #7: <unknown function> + 0x8609 (0x7f9a84eb4609 in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #8: clone + 0x43 (0x7f9a84c7f353 in /lib/x86_64-linux-gnu/libc.so.6)
```