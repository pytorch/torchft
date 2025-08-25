"""
This is a file for TorchX components used for testing torchft.
"""

import os
from typing import Dict, Optional

import torchx.specs as specs


def hsdp(
    *script_args: str,
    replicas: int = 2,
    workers_per_replica: int = 1,
    max_restarts: int = 10,
    script: str = "train_ddp.py",
    env: Optional[Dict[str, str]] = None,
    image: str = "",
    h: Optional[str] = None,
    cpu: int = 2,
    gpu: int = 0,
    memMB: int = 1024,
) -> specs.AppDef:
    assert replicas > 0, "replicas must be > 0"
    assert workers_per_replica > 0, "workers_per_replica must be > 0"

    env = env or {}

    # Enable logging for PyTorch, torchelastic and Rust.
    env.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
    env.setdefault("LOGLEVEL", "INFO")
    env.setdefault("RUST_BACKTRACE", "1")

    # Enable colored logging for torchft Rust logger.
    env.setdefault("CLICOLOR_FORCE", "1")

    # Set lighthouse address for replicas
    # This must be run externally
    env.setdefault(
        "TORCHFT_LIGHTHOUSE",
        os.environ.get("TORCHFT_LIGHTHOUSE", f"http://localhost:29510"),
    )

    # Disable CUDA for CPU-only jobs
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Disable XPU for CPU-only jobs
    env.setdefault("XPU_VISIBLE_DEVICES", "")

    roles = []
    for replica_id in range(replicas):
        cmd = [
            f"--master_port={29600+replica_id}",
            "--nnodes=1",
            f"--nproc_per_node={workers_per_replica}",
            f"--max_restarts={max_restarts}",
        ]
        if script:
            cmd += [script]
        cmd += list(script_args)

        roles.append(
            specs.Role(
                name=f"replica_{replica_id}",
                image=image,
                min_replicas=workers_per_replica,
                num_replicas=workers_per_replica,
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                max_retries=0,
                env={
                    "REPLICA_GROUP_ID": str(replica_id),
                    "NUM_REPLICA_GROUPS": str(replicas),
                    **env,
                },
                entrypoint="torchrun",
                args=cmd,
            )
        )

    return specs.AppDef(
        name="torchft",
        roles=roles,
    )
