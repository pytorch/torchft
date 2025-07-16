from dataclasses import dataclass
from datetime import timedelta
from typing import Hashable, List, Optional

class ManagerClient:
    def __init__(self, addr: str, connect_timeout: timedelta) -> None: ...
    def _quorum(
        self,
        group_rank: int,
        step: int,
        checkpoint_metadata: str,
        shrink_only: bool,
        timeout: timedelta,
        commit_failures: int,
        init_sync: bool = True,
    ) -> QuorumResult: ...
    def _checkpoint_metadata(self, rank: int, timeout: timedelta) -> str: ...
    def should_commit(
        self,
        group_rank: int,
        step: int,
        should_commit: bool,
        timeout: timedelta,
    ) -> bool: ...

class QuorumResult:
    quorum_id: int
    replica_rank: int
    replica_world_size: int
    recover_src_manager_address: str
    recover_src_replica_rank: Optional[int]
    recover_dst_replica_ranks: List[int]
    store_address: str
    max_step: int
    max_replica_rank: Optional[int]
    max_world_size: int
    heal: bool
    commit_failures: int

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
        quorum_retries: int,
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

@dataclass
class QuorumMember:
    replica_id: str
    address: str
    store_address: str
    step: int
    world_size: int
    shrink_only: bool
    data: Optional[dict[Hashable, object]] = None

@dataclass
class Timestamp:
    seconds: int
    nanos: int

@dataclass
class Quorum:
    quorum_id: str
    participants: List[QuorumMember]
    created: Timestamp

@dataclass
class LighthouseClient:
    addr: str
    connect_timeout: timedelta

    def quorum(
        self,
        replica_id: str,
        timeout: timedelta,
        address: Optional[str] = None,
        store_address: Optional[str] = None,
        step: Optional[int] = None,
        world_size: Optional[int] = None,
        shrink_only: Optional[bool] = None,
        data: Optional[dict[Hashable, object]] = None,
    ) -> Quorum: ...
    def heartbeat(
        self,
        replica_id: str,
        timeout: timedelta = timedelta(seconds=5),
    ) -> None: ...
