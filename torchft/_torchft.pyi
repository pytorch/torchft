from datetime import timedelta
from typing import List, Optional

class ManagerClient:
    def __init__(self, addr: str, connect_timeout: timedelta) -> None: ...
    def _quorum(
        self,
        rank: int,
        step: int,
        checkpoint_metadata: str,
        shrink_only: bool,
        timeout: timedelta,
    ) -> QuorumResult: ...
    def _checkpoint_metadata(self, rank: int, timeout: timedelta) -> str: ...
    def should_commit(
        self,
        rank: int,
        step: int,
        should_commit: bool,
        timeout: timedelta,
    ) -> bool: ...

class QuorumResult:
    quorum_id: int
    replica_rank: int
    replica_world_size: int
    recover_src_manager_address: str
    recover_src_rank: Optional[int]
    recover_dst_ranks: List[int]
    store_address: str
    max_step: int
    max_rank: Optional[int]
    max_world_size: int
    heal: bool

class ManagerServer:
    def __init__(
        self,
        replica_id: str,
        lighthouse_addr: str,
        hostname: str,
        bind: str,
        store_addr: str,
        world_size: int,
        heartbeat_interval: timedelta,
        connect_timeout: timedelta,
    ) -> None: ...
    def address(self) -> str: ...
    def shutdown(self) -> None: ...

class LighthouseServer:
    def __init__(
        self,
        bind: str,
        min_replicas: int,
        join_timeout_ms: Optional[int] = None,
        quorum_tick_ms: Optional[int] = None,
        heartbeat_timeout_ms: Optional[int] = None,
    ) -> None: ...
    def address(self) -> str: ...
    def shutdown(self) -> None: ...
