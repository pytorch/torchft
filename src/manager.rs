// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use core::net::SocketAddr;
use std::collections::HashMap;
use std::collections::HashSet;
#[cfg(test)]
use std::println as info;
#[cfg(test)]
use std::println as warn;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
#[cfg(not(test))]
use log::info;
#[cfg(not(test))]
use log::warn;
use tokio::sync::Mutex;
use tokio::sync::broadcast;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Channel;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

use crate::net::connect;
use crate::timeout::try_parse_grpc_timeout;
use crate::torchftpb::CheckpointMetadataRequest;
use crate::torchftpb::CheckpointMetadataResponse;
use crate::torchftpb::KillRequest;
use crate::torchftpb::KillResponse;
use crate::torchftpb::LighthouseHeartbeatRequest;
use crate::torchftpb::LighthouseQuorumRequest;
use crate::torchftpb::LighthouseQuorumResponse;
use crate::torchftpb::ManagerQuorumRequest;
use crate::torchftpb::ManagerQuorumResponse;
use crate::torchftpb::Quorum;
use crate::torchftpb::QuorumMember;
use crate::torchftpb::ShouldCommitRequest;
use crate::torchftpb::ShouldCommitResponse;
use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;
use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::manager_service_server::ManagerService;
use crate::torchftpb::manager_service_server::ManagerServiceServer;

// The replica_id string is of the form {replica_name}:{uuid} or just {uuid} (see torchft/manager.py)
// We can parse the replica_id if it exists, otherwise we just use the uuid
macro_rules! info_with_replica {
    ($replica_id:expr, $($arg:tt)*) => {{
        let parts: Vec<&str> = $replica_id.splitn(2, ':').collect();
        if parts.len() == 2 {
            // If there are two parts, use the replica name
            info!("[Replica {}] {}", parts[0], format!($($arg)*))
        } else {
            // Otherwise, just use the UUID
            info!("[Replica {}] {}", $replica_id, format!($($arg)*))
        };
    }};
}

struct ManagerState {
    checkpoint_metadata: HashMap<i64, String>,
    channel: broadcast::Sender<Quorum>,
    participants: HashMap<i64, QuorumMember>,

    should_commit_channel: broadcast::Sender<bool>,
    should_commit_failures: HashSet<i64>,
    should_commit_count: HashSet<i64>,

    lighthouse_client: LighthouseServiceClient<Channel>,
}

pub struct Manager {
    replica_id: String,
    hostname: String,
    store_address: String,
    world_size: u64,
    state: Mutex<ManagerState>,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
    heartbeat_interval: Duration,
    lighthouse_addr: String,
    connect_timeout: Duration,
    quorum_retries: i64,
}

pub async fn manager_client_new(
    addr: String,
    connect_timeout: Duration,
) -> Result<ManagerServiceClient<Channel>> {
    info!(
        "Creating ManagerClient: establishing connection to {}",
        &addr
    );
    let conn = connect(addr, connect_timeout).await?;
    Ok(ManagerServiceClient::new(conn))
}

pub async fn lighthouse_client_new(
    addr: String,
    connect_timeout: Duration,
) -> Result<LighthouseServiceClient<Channel>> {
    info!(
        "Creating LighthouseClient: establishing connection to {}",
        &addr
    );
    let conn = connect(addr, connect_timeout).await?;
    Ok(LighthouseServiceClient::new(conn))
}

impl Manager {
    pub async fn new(
        replica_id: String,
        lighthouse_addr: String,
        hostname: String,
        bind: String,
        store_addr: String,
        world_size: u64,
        heartbeat_interval: Duration,
        connect_timeout: Duration,
        quorum_retries: i64,
    ) -> Result<Arc<Self>> {
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        let local_addr = listener.local_addr()?;

        let (should_commit_tx, _) = broadcast::channel(16);
        let (tx, _) = broadcast::channel(16);

        let client = lighthouse_client_new(lighthouse_addr.clone(), connect_timeout).await?;

        Ok(Arc::new(Self {
            replica_id,
            lighthouse_addr,
            connect_timeout,
            hostname,
            store_address: store_addr,
            world_size,
            heartbeat_interval,
            state: Mutex::new(ManagerState {
                checkpoint_metadata: HashMap::new(),
                channel: tx,
                participants: HashMap::new(),

                should_commit_channel: should_commit_tx,
                should_commit_count: HashSet::new(),
                should_commit_failures: HashSet::new(),

                lighthouse_client: client,
            }),
            local_addr,
            listener: Mutex::new(Some(listener)),
            quorum_retries,
        }))
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_heartbeat());

        set.spawn(self.clone()._run_grpc());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }

    pub fn address(&self) -> String {
        format!("http://{}:{}", self.hostname, self.local_addr.port())
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info_with_replica!(self.replica_id, "Manager listening on {}", self.address());

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        Server::builder()
            .add_service(ManagerServiceServer::new(self))
            .serve_with_incoming(incoming)
            .await
            .map_err(|e| e.into())
    }

    async fn _run_heartbeat(self: Arc<Self>) -> Result<()> {
        loop {
            let mut client = {
                let state = self.state.lock().await;
                state.lighthouse_client.clone()
            };

            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: self.replica_id.clone(),
            });

            if let Err(e) = client.heartbeat(request).await {
                info_with_replica!(
                    self.replica_id,
                    "Failed to send heartbeat to lighthouse: {}",
                    e.to_string()
                );
                let _ = self.create_lighthouse_client().await;
            }

            sleep(self.heartbeat_interval).await;
        }
    }

    async fn _run_quorum(
        self: Arc<Self>,
        requester: QuorumMember,
        timeout: Duration,
    ) -> Result<(), Status> {
        info_with_replica!(self.replica_id, "All workers joined - starting quorum");

        let lighthouse_request = LighthouseQuorumRequest {
            requester: Some(requester),
        };

        let response = self
            ._quorum_with_retries(timeout, lighthouse_request)
            .await?;

        let resp = response.into_inner();

        info_with_replica!(self.replica_id, "got lighthouse quorum {:?}", resp);

        let state = self.state.lock().await;
        // TODO: We don't broadcast in cases when this method returns an error, resulting in a hang
        state
            .channel
            .send(
                resp.quorum
                    .ok_or_else(|| Status::internal("missing quorum"))?,
            )
            .map_err(|e| Status::from_error(e.into()))?;

        Ok(())
    }

    async fn _quorum_with_retries(
        &self,
        timeout: Duration,
        lighthouse_request: LighthouseQuorumRequest,
    ) -> Result<tonic::Response<LighthouseQuorumResponse>, Status> {
        let mut retry_count = 0;
        loop {
            let mut client = {
                let state = self.state.lock().await;
                state.lighthouse_client.clone()
            };

            let mut request = tonic::Request::new(lighthouse_request.clone());
            request.set_timeout(timeout);

            let result = tokio::time::timeout(timeout, client.quorum(request)).await;

            let mut sleep_time = 100;
            match result {
                Ok(Ok(response)) => {
                    return Ok(response);
                }
                Ok(Err(e)) => {
                    info_with_replica!(
                        self.replica_id,
                        "lighthouse quorum failed. error: {}",
                        e.to_string()
                    );
                    sleep_time = timeout.as_millis() as u64
                        / (std::cmp::max(self.quorum_retries + 1, 1 as i64)) as u64;
                    sleep_time = std::cmp::max(100, sleep_time);
                }
                Err(e) => {
                    info_with_replica!(
                        self.replica_id,
                        "lighthouse quorum timeout. error: {}",
                        e.to_string()
                    );
                }
            }

            if retry_count == self.quorum_retries {
                return Err(Status::internal(format!(
                    "lighthouse quorum failed after {} retries.",
                    retry_count,
                )));
            }

            tokio::time::sleep(Duration::from_millis(sleep_time)).await;

            // Reset the client since the lighthouse server might have failed
            // If this also fails, consider increasing `connect_timeout`.
            let _ = self.create_lighthouse_client().await;

            retry_count += 1;
        }
    }

    async fn create_lighthouse_client(&self) -> Result<(), Status> {
        // Reset the client since the lighthouse server might have failed
        // If this also fails, consider increasing `connect_timeout`.
        let lighthouse_client =
            lighthouse_client_new(self.lighthouse_addr.clone(), self.connect_timeout).await;

        match lighthouse_client {
            Ok(client) => {
                let mut state = self.state.lock().await;
                state.lighthouse_client = client;
                return Ok(());
            }
            Err(e) => {
                return Err(Status::internal(format!(
                    "Failed to connect to lighthouse. error: {}",
                    e.to_string(),
                )));
            }
        }
    }
}

#[tonic::async_trait]
impl ManagerService for Arc<Manager> {
    async fn quorum(
        &self,
        request: Request<ManagerQuorumRequest>,
    ) -> Result<Response<ManagerQuorumResponse>, Status> {
        let req = request.get_ref();
        let group_rank = req.group_rank;

        info_with_replica!(
            self.replica_id,
            "Start quorum for group_rank {}",
            group_rank
        );

        let timeout = try_parse_grpc_timeout(&request.metadata())
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "invalid timeout {}",
                    e.to_str().unwrap_or("invalid")
                ))
            })?
            .ok_or_else(|| Status::invalid_argument("missing timeout"))?;

        let mut rx = {
            let mut state = self.state.lock().await;

            // save checkpoint server info for healing process
            // TODO: make separate call to set?
            state
                .checkpoint_metadata
                .insert(req.group_rank, req.checkpoint_metadata.clone());

            let member = QuorumMember {
                replica_id: self.replica_id.clone(),
                address: self.address(),
                store_address: self.store_address.clone(),
                step: req.step,
                world_size: self.world_size,
                shrink_only: req.shrink_only,
                data: String::new(),
                commit_failures: req.commit_failures,
            };
            // TODO check step
            state.participants.insert(group_rank, member.clone());
            let rx = state.channel.subscribe();

            if (state.participants.len() as u64) == self.world_size {
                state.participants.clear();
                let self_clone = self.clone();
                tokio::spawn(async move {
                    let _ = self_clone._run_quorum(member, timeout).await;
                });
            }

            rx
        };

        let quorum = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        info_with_replica!(
            self.replica_id,
            "Finished quorum for group_rank {}",
            group_rank
        );

        let reply = compute_quorum_results(&self.replica_id, group_rank, &quorum, req.init_sync)?;

        Ok(Response::new(reply))
    }

    async fn checkpoint_metadata(
        &self,
        request: Request<CheckpointMetadataRequest>,
    ) -> Result<Response<CheckpointMetadataResponse>, Status> {
        let state = self.state.lock().await;

        let req = request.into_inner();

        let metadata = state
            .checkpoint_metadata
            .get(&req.rank)
            .ok_or_else(|| Status::invalid_argument("rank not found"))?;

        let reply = CheckpointMetadataResponse {
            checkpoint_metadata: metadata.clone(),
        };
        Ok(Response::new(reply))
    }

    async fn should_commit(
        &self,
        request: Request<ShouldCommitRequest>,
    ) -> Result<Response<ShouldCommitResponse>, Status> {
        let req = request.into_inner();
        let group_rank = req.group_rank;

        info_with_replica!(
            self.replica_id,
            "should_commit request from {} should_commit={}",
            group_rank,
            req.should_commit
        );

        // TODO: check step count

        let mut rx = {
            let mut state = self.state.lock().await;

            if !req.should_commit {
                state.should_commit_failures.insert(group_rank);
            }
            state.should_commit_count.insert(group_rank);

            let rx = state.should_commit_channel.subscribe();

            if state.should_commit_count.len() == self.world_size as usize {
                let decision = state.should_commit_failures.len() == 0;
                info_with_replica!(
                    self.replica_id,
                    "should_commit completed should_commit={}",
                    decision
                );

                state
                    .should_commit_channel
                    .send(decision)
                    .map_err(|e| Status::from_error(e.into()))?;

                // reset state
                state.should_commit_count.clear();
                state.should_commit_failures.clear();
                let (should_commit_tx, _) = broadcast::channel(16);
                state.should_commit_channel = should_commit_tx;
            }

            rx
        };

        let should_commit = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let reply = ShouldCommitResponse { should_commit };
        Ok(Response::new(reply))
    }

    async fn kill(&self, request: Request<KillRequest>) -> Result<Response<KillResponse>, Status> {
        let req = request.into_inner();

        warn!("got kill request: {}", req.msg);
        std::process::exit(1);
    }
}

fn compute_quorum_results(
    replica_id: &str,
    group_rank: i64,
    quorum: &Quorum,
    init_sync: bool,
) -> Result<ManagerQuorumResponse, Status> {
    let mut participants = quorum.participants.clone();
    participants.sort_by(|a, b| a.replica_id.cmp(&b.replica_id));

    // Compute the rank of the replica in the returned quorum.
    let replica_rank = participants
        .iter()
        .enumerate()
        .find_map(|(i, p)| {
            if p.replica_id == replica_id {
                Some(i)
            } else {
                None
            }
        })
        .ok_or_else(|| {
            Status::not_found(format!(
                "replica {} not participating in returned quorum",
                replica_id
            ))
        })?;

    let step = participants[replica_rank].step;

    // Compute the details for workers at max step.
    let max_step = participants.iter().map(|p| p.step).max().unwrap();
    let max_participants: Vec<&QuorumMember> =
        participants.iter().filter(|p| p.step == max_step).collect();
    let max_replica_rank = max_participants.iter().enumerate().find_map(|(i, p)| {
        if p.replica_id == replica_id {
            Some(i as i64)
        } else {
            None
        }
    });

    // The primary TCPStore to use for this rank.
    // There is one TCPStore per replica.
    let primary_replica_rank = group_rank as usize % max_participants.len();
    let primary = max_participants[primary_replica_rank];

    // Compute recovery assignments

    let force_recover = init_sync && max_step == 0;

    // Nodes are recovering if
    // 1. not at the max step (init_sync)
    // 2. max_step == 0 and not the primary replica
    let all_recover_dst_replica_ranks: Vec<usize> = participants
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p.step != max_step || force_recover && primary.replica_id != p.replica_id {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let all_recover_dst_replica_ranks_set =
        all_recover_dst_replica_ranks.iter().collect::<HashSet<_>>();
    let up_to_date_ranks: Vec<usize> = participants
        .iter()
        .enumerate()
        .filter_map(|(i, _p)| {
            if !all_recover_dst_replica_ranks_set.contains(&i) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // This is a map of rank to the ranks that are recovering from that node.
    let mut recovery_assignments: HashMap<usize, Vec<i64>> = HashMap::new();
    // The rank of the node that this rank is recovering from.
    let mut recover_src_replica_rank: Option<i64> = None;
    for (i, recovering_rank) in all_recover_dst_replica_ranks.iter().enumerate() {
        let up_to_date_idx = (i + group_rank as usize) % up_to_date_ranks.len();
        let recovering_recover_src_replica_rank = up_to_date_ranks[up_to_date_idx];
        if !recovery_assignments.contains_key(&recovering_recover_src_replica_rank) {
            recovery_assignments.insert(recovering_recover_src_replica_rank, Vec::new());
        }
        recovery_assignments
            .get_mut(&recovering_recover_src_replica_rank)
            .unwrap()
            .push(*recovering_rank as i64);
        if *recovering_rank == replica_rank {
            recover_src_replica_rank = Some(recovering_recover_src_replica_rank as i64);
        }
    }

    let heal = recover_src_replica_rank.is_some();
    if heal {
        info_with_replica!(
            replica_id,
            "healing is required step={}, max_step={}, recover_src_replica_rank={}",
            step,
            max_step,
            recover_src_replica_rank.unwrap()
        );
    }

    let recover_src_manager_address = match recover_src_replica_rank {
        Some(r) => participants[r as usize].address.clone(),
        None => "".to_string(),
    };

    Ok(ManagerQuorumResponse {
        quorum_id: quorum.quorum_id,
        // address is used for looking up the checkpoint server address.
        recover_src_manager_address,
        recover_src_replica_rank,
        recover_dst_replica_ranks: recovery_assignments
            .get(&replica_rank)
            .map_or_else(Vec::new, |v| v.clone()),
        store_address: primary.store_address.clone(),
        max_step,
        max_replica_rank,
        max_world_size: max_participants.len() as i64,
        replica_rank: replica_rank as i64,
        replica_world_size: participants.len() as i64,
        heal,
        commit_failures: participants
            .iter()
            .map(|p| p.commit_failures)
            .max()
            .unwrap_or(0),
        replica_ids: participants.iter().map(|p| p.replica_id.clone()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use tokio::net::TcpListener;
    use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;

    use super::*;
    use crate::lighthouse::Lighthouse;
    use crate::lighthouse::LighthouseOpt;
    use crate::torchftpb::LighthouseHeartbeatResponse;
    use crate::torchftpb::lighthouse_service_server::LighthouseService;
    use crate::torchftpb::lighthouse_service_server::LighthouseServiceServer;

    async fn should_commit(group_rank: i64, should_commit: bool) -> Result<ShouldCommitResponse> {
        let mut client = manager_client_new(
            "http://localhost:29531".to_string(),
            Duration::from_secs(10),
        )
        .await?;

        let request = tonic::Request::new(ShouldCommitRequest {
            group_rank,
            step: 1,
            should_commit,
        });
        let resp = client.should_commit(request).await?;

        Ok(resp.into_inner())
    }

    #[tokio::test]
    async fn test_should_commit() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 1,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let manager = Manager::new(
            "rep_id".to_string(),
            lighthouse.address(),
            "addr".to_string(),
            "[::]:29531".to_string(),
            "store_addr".to_string(),
            2,                          // world size
            Duration::from_millis(100), // heartbeat interval
            Duration::from_secs(10),    // connect timeout
            0,                          // quorum retries
        )
        .await?;
        let manager_fut = tokio::spawn(manager._run_grpc());

        let fut_a = tokio::spawn(should_commit(0, true));
        let fut_b = tokio::spawn(should_commit(1, true));
        let resp_a = fut_a.await??;
        let resp_b = fut_b.await??;

        assert!(resp_a.should_commit);
        assert!(resp_b.should_commit);

        let fut_a = tokio::spawn(should_commit(0, true));
        let fut_b = tokio::spawn(should_commit(1, false));
        let resp_a = fut_a.await??;
        let resp_b = fut_b.await??;

        assert!(!resp_a.should_commit);
        assert!(!resp_b.should_commit);

        manager_fut.abort();
        lighthouse_fut.abort();

        Ok(())
    }

    #[tokio::test]
    async fn test_get_quorum() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 1,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let manager = Manager::new(
            "rep_id".to_string(),
            lighthouse.address(),
            "localhost".to_string(),
            "[::]:0".to_string(),
            "store_addr".to_string(),
            1,                          // world size
            Duration::from_millis(100), // heartbeat interval
            Duration::from_secs(10),    // connect timeout
            0,                          // quorum retries
        )
        .await?;
        let manager_fut = tokio::spawn(manager.clone().run());

        let mut client = manager_client_new(manager.address(), Duration::from_secs(10)).await?;

        let mut request = tonic::Request::new(ManagerQuorumRequest {
            group_rank: 0,
            step: 123,
            checkpoint_metadata: "addr".to_string(),
            shrink_only: false,
            init_sync: true,
            commit_failures: 3,
        });
        request.set_timeout(Duration::from_secs(10));
        let resp = client.quorum(request).await?.into_inner();

        manager_fut.abort();
        lighthouse_fut.abort();

        assert_eq!(resp.quorum_id, 1);
        assert_eq!(resp.recover_src_manager_address, "".to_string());
        assert_eq!(resp.store_address, "store_addr".to_string());
        assert_eq!(resp.max_step, 123);
        assert_eq!(resp.max_replica_rank, Some(0));
        assert_eq!(resp.max_world_size, 1);
        assert_eq!(resp.replica_rank, 0);
        assert_eq!(resp.replica_world_size, 1);
        assert_eq!(resp.heal, false);
        assert_eq!(resp.commit_failures, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_quorum_heal_first_step() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 2,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let mut manager_futs: Vec<tokio::task::JoinHandle<Result<ManagerQuorumResponse>>> =
            Vec::new();

        for replica_id in 0..2 {
            let lighthouse_addr = lighthouse.address();
            manager_futs.push(tokio::spawn(async move {
                let manager = Manager::new(
                    format!("rep_{}", replica_id),
                    lighthouse_addr,
                    "localhost".to_string(),
                    "[::]:0".to_string(),
                    "store_addr".to_string(),
                    1,                          // world size
                    Duration::from_millis(100), // heartbeat interval
                    Duration::from_secs(10),    // connect timeout
                    0,                          // quorum retries
                )
                .await?;
                let manager_fut = tokio::spawn(manager.clone().run());

                let mut client =
                    manager_client_new(manager.address(), Duration::from_secs(10)).await?;

                let mut request = tonic::Request::new(ManagerQuorumRequest {
                    group_rank: 0,
                    step: 0,
                    checkpoint_metadata: "addr".to_string(),
                    shrink_only: false,
                    init_sync: true,
                    commit_failures: 0,
                });
                request.set_timeout(Duration::from_secs(10));

                let result = client.quorum(request).await?.into_inner();

                manager_fut.abort();

                Ok(result)
            }));
        }

        let resp_a = manager_futs.swap_remove(0).await??;
        let resp_b = manager_futs.swap_remove(0).await??;

        lighthouse_fut.abort();

        assert_eq!(resp_a.quorum_id, 1);
        assert_eq!(resp_a.max_step, 0);
        assert_eq!(resp_a.replica_rank, 0);
        assert_eq!(resp_a.replica_world_size, 2);
        assert_eq!(resp_a.heal, false);

        assert_eq!(resp_b.quorum_id, 1);
        assert_eq!(resp_b.max_step, 0);
        assert_eq!(resp_b.replica_rank, 1);
        assert_eq!(resp_b.replica_world_size, 2);
        assert_eq!(resp_b.heal, true);

        Ok(())
    }

    #[tokio::test]
    async fn test_checkpoint_metadata() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 1,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let manager = Manager::new(
            "rep_id".to_string(),
            lighthouse.address(),
            "localhost".to_string(),
            "[::]:0".to_string(),
            "store_addr".to_string(),
            1,                          // world size
            Duration::from_millis(100), // heartbeat interval
            Duration::from_secs(10),    // connect timeout
            0,                          // quorum retries
        )
        .await?;
        let manager_fut = tokio::spawn(manager.clone().run());

        let mut client = manager_client_new(manager.address(), Duration::from_secs(10)).await?;

        let request = tonic::Request::new(CheckpointMetadataRequest { rank: 0 });
        let resp = client.checkpoint_metadata(request).await;
        assert!(resp.err().unwrap().to_string().contains("rank not found"));

        {
            let mut state = manager.state.lock().await;

            state.checkpoint_metadata.insert(0, "addr".to_string());
        }

        let request = tonic::Request::new(CheckpointMetadataRequest { rank: 0 });
        let resp = client.checkpoint_metadata(request).await?.into_inner();
        assert_eq!(resp.checkpoint_metadata, "addr".to_string());

        manager_fut.abort();
        lighthouse_fut.abort();

        Ok(())
    }

    #[tokio::test]
    async fn test_compute_quorum_results_first_step() -> Result<()> {
        let quorum = Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "replica_0".to_string(),
                    address: "addr_0".to_string(),
                    store_address: "store_addr_0".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_1".to_string(),
                    address: "addr_1".to_string(),
                    store_address: "store_addr_1".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: None,
        };

        // rank 0

        let results = compute_quorum_results("replica_0", 0, &quorum, true)?;
        assert!(!results.heal);
        assert_eq!(results.replica_rank, 0);
        assert_eq!(results.recover_src_replica_rank, None);
        assert_eq!(results.recover_dst_replica_ranks, vec![1]);

        let results = compute_quorum_results("replica_1", 0, &quorum, true)?;
        assert!(results.heal);
        assert_eq!(results.replica_rank, 1);
        assert_eq!(results.recover_src_replica_rank, Some(0));
        assert_eq!(results.recover_dst_replica_ranks, Vec::<i64>::new());

        // rank 1 assignments should be offset from rank 0 above and the primary

        let results = compute_quorum_results("replica_1", 1, &quorum, true)?;
        assert!(!results.heal);
        assert_eq!(results.replica_rank, 1);
        assert_eq!(results.recover_src_replica_rank, None);
        assert_eq!(results.recover_dst_replica_ranks, vec![0]);

        Ok(())
    }

    #[tokio::test]
    async fn test_compute_quorum_results_recovery() -> Result<()> {
        let quorum = Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "replica_0".to_string(),
                    address: "addr_0".to_string(),
                    store_address: "store_addr_0".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_1".to_string(),
                    address: "addr_1".to_string(),
                    store_address: "store_addr_1".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_2".to_string(),
                    address: "addr_2".to_string(),
                    store_address: "store_addr_2".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_3".to_string(),
                    address: "addr_3".to_string(),
                    store_address: "store_addr_3".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_4".to_string(),
                    address: "addr_4".to_string(),
                    store_address: "store_addr_4".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: None,
        };

        // rank 0

        let results = compute_quorum_results("replica_0", 0, &quorum, true)?;
        assert!(results.heal);
        assert_eq!(results.recover_src_manager_address, "addr_1".to_string());
        assert_eq!(results.replica_rank, 0);
        assert_eq!(results.recover_src_replica_rank, Some(1));
        assert!(results.recover_dst_replica_ranks.is_empty());

        let results = compute_quorum_results("replica_1", 0, &quorum, true)?;
        assert!(!results.heal);
        assert_eq!(results.recover_src_manager_address, "".to_string());
        assert_eq!(results.replica_rank, 1);
        assert_eq!(results.recover_src_replica_rank, None);
        assert_eq!(results.recover_dst_replica_ranks, vec![0, 4]);

        let results = compute_quorum_results("replica_3", 0, &quorum, true)?;
        assert!(!results.heal);
        assert_eq!(results.replica_rank, 3);
        assert_eq!(results.recover_src_replica_rank, None);
        assert_eq!(results.recover_dst_replica_ranks, vec![2]);

        // rank 1 assignments should be offset from rank 0 above

        let results = compute_quorum_results("replica_1", 1, &quorum, true)?;
        assert!(!results.heal);
        assert_eq!(results.replica_rank, 1);
        assert_eq!(results.recover_src_replica_rank, None);
        assert_eq!(results.recover_dst_replica_ranks, vec![2]);

        Ok(())
    }

    #[tokio::test]
    async fn test_compute_quorum_results_skip_init_sync() -> Result<()> {
        let mut quorum = Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "replica_0".to_string(),
                    address: "addr_0".to_string(),
                    store_address: "store_addr_0".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_1".to_string(),
                    address: "addr_1".to_string(),
                    store_address: "store_addr_1".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: None,
        };

        // baseline w/ init_sync=true
        let results = compute_quorum_results("replica_0", 0, &quorum, true)?;
        assert!(!results.heal);

        let results = compute_quorum_results("replica_1", 0, &quorum, true)?;
        assert!(results.heal);

        // init_sync=false
        let results = compute_quorum_results("replica_1", 0, &quorum, false)?;
        assert!(!results.heal);

        // init_sync=false, step=1
        quorum.participants[0].step = 1;
        let results = compute_quorum_results("replica_1", 0, &quorum, false)?;
        assert!(results.heal);

        Ok(())
    }

    #[tokio::test]
    async fn test_compute_quorum_results_commit_failures() -> Result<()> {
        let quorum = Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "replica_0".to_string(),
                    address: "addr_0".to_string(),
                    store_address: "store_addr_0".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica_1".to_string(),
                    address: "addr_1".to_string(),
                    store_address: "store_addr_1".to_string(),
                    step: 0,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 2,
                },
            ],
            created: None,
        };

        let results = compute_quorum_results("replica_0", 0, &quorum, true)?;
        assert_eq!(results.commit_failures, 2);

        Ok(())
    }

    #[derive(Default)]
    pub struct MockLighthouse {
        request_count: Arc<Mutex<usize>>,
        fail_count: usize,
    }

    impl MockLighthouse {
        pub fn new(fail_count: usize) -> Self {
            Self {
                request_count: Arc::new(Mutex::new(0)),
                fail_count,
            }
        }
    }

    #[tonic::async_trait]
    impl LighthouseService for MockLighthouse {
        async fn quorum(
            &self,
            request: Request<LighthouseQuorumRequest>,
        ) -> Result<Response<LighthouseQuorumResponse>, Status> {
            let mut count = self.request_count.lock().await;
            *count += 1;
            let current_count = *count;

            if current_count <= self.fail_count {
                return Err(Status::internal(format!(
                    "simulated failure (request {}/{})",
                    current_count, self.fail_count
                )));
            }

            let requester = request.into_inner().requester.unwrap_or_default();

            Ok(Response::new(LighthouseQuorumResponse {
                quorum: Some(Quorum {
                    participants: vec![requester],
                    quorum_id: 1,
                    created: None,
                }),
            }))
        }

        async fn heartbeat(
            &self,
            _request: Request<LighthouseHeartbeatRequest>,
        ) -> Result<Response<LighthouseHeartbeatResponse>, Status> {
            Ok(Response::new(LighthouseHeartbeatResponse {}))
        }
    }

    pub async fn start_mock_lighthouse(
        fail_count: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("[::]:0").await?;
        let addr = listener.local_addr()?;
        let mock_service = LighthouseServiceServer::new(MockLighthouse::new(fail_count));
        let incoming = TcpListenerStream::new(listener);
        tokio::spawn(async move {
            Server::builder()
                .add_service(mock_service)
                .serve_with_incoming(incoming)
                .await
                .expect("Mock Lighthouse server failed to start");
        });
        Ok(format!(
            "http://{}:{}",
            gethostname::gethostname().into_string().unwrap(),
            addr.port()
        ))
    }

    #[tokio::test]
    async fn test_get_quorum_when_lighthouse_down() -> Result<()> {
        let addr = start_mock_lighthouse(1).await.unwrap();

        let manager = Manager::new(
            "rep_id".to_string(),
            addr,
            "localhost".to_string(),
            "[::]:0".to_string(),
            "store_addr".to_string(),
            1,
            Duration::from_millis(100),
            Duration::from_secs(10),
            1,
        )
        .await?;
        let manager_fut = tokio::spawn(manager.clone().run());

        let mut client = manager_client_new(manager.address(), Duration::from_secs(3)).await?;

        let mut request = tonic::Request::new(ManagerQuorumRequest {
            group_rank: 0,
            step: 123,
            checkpoint_metadata: "addr".to_string(),
            shrink_only: false,
            init_sync: true,
            commit_failures: 3,
        });
        request.set_timeout(Duration::from_secs(3));
        let resp = client.quorum(request).await?.into_inner();

        manager_fut.abort();

        assert_eq!(resp.quorum_id, 1);

        Ok(())
    }
}
