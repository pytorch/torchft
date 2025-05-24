// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use core::net::SocketAddr;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use std::time::Duration;
use std::time::{Instant, SystemTime};

use crate::torchftpb::FailureNotification;
use anyhow::{anyhow, Result};
use askama::Template;
use axum::{
    extract::Path,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use chrono;
use gethostname::gethostname;
use log::{error, info, warn};
use serde_json;
use std::fs;
use structopt::StructOpt;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::interval;
use tokio_stream::wrappers::{
    errors::BroadcastStreamRecvError as TokioStreamBroadcastStreamRecvError, BroadcastStream,
};
use tokio_stream::StreamExt;
use tonic::service::Routes;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use futures_core::Stream;
use std::pin::Pin;

use crate::manager::manager_client_new;
use crate::torchftpb::{
    lighthouse_service_server::{LighthouseService, LighthouseServiceServer},
    KillRequest, LighthouseConfigRequest, LighthouseConfigResponse, LighthouseHeartbeatRequest,
    LighthouseHeartbeatResponse, LighthouseQuorumRequest, LighthouseQuorumResponse, Quorum,
    QuorumMember, SubscribeFailuresRequest,
};

use serde::Deserialize;

#[derive(Clone)]
struct QuorumMemberDetails {
    joined: Instant,
    member: QuorumMember,
}

struct State {
    quorum_channel: broadcast::Sender<Quorum>,
    // Tracks currently active participants in the process of forming a quorum.
    // Replicas are added upon receiving a `LighthouseQuorumRequest`.
    // Replicas are cleared after a quorum is successfully formed OR
    // removed by `_failure_tick` if their heartbeat expires.
    participants: HashMap<String, QuorumMemberDetails>,
    prev_quorum: Option<Quorum>,
    quorum_id: i64,

    // Stores the last heartbeat time for each replica ID.
    // Replicas are added/updated upon receiving `LighthouseHeartbeatRequest` or `LighthouseQuorumRequest`.
    // Replicas are removed by `_failure_tick` if their heartbeat expires and a failure notification is sent.
    heartbeats: HashMap<String, Instant>,

    // Stores the timestamp of when a replica was first detected as failed (heartbeat expired).
    // This is used to ensure only one `FailureNotification` is sent per failure event.
    // Replicas are added by `_failure_tick` upon detecting a new failure.
    // Replicas are removed by `_failure_tick` if a subsequent heartbeat is received (signifying recovery).
    failures: HashMap<String, Instant>,

    // Broadcast channel for sending failure notifications to subscribers.
    pub failure_channel: broadcast::Sender<FailureNotification>,

    // Configuration data as serde_json::Map (loaded from config file if provided)
    config_data: serde_json::Map<String, serde_json::Value>,
}

pub struct Lighthouse {
    state: Mutex<State>,
    opt: LighthouseOpt,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
    change_logger: ChangeLogger,
}

struct ChangeLogger {
    last_reason: std::sync::Mutex<Option<String>>,
}
impl ChangeLogger {
    fn new() -> Self {
        ChangeLogger {
            last_reason: std::sync::Mutex::new(None),
        }
    }
    fn log_if_changed(&self, reason: &str) {
        let mut last_reason = self.last_reason.lock().unwrap();
        if last_reason.as_deref() != Some(reason) {
            info!("Quorum status: {}", reason);
            *last_reason = Some(reason.to_string());
        }
    }
}

#[derive(StructOpt, Debug, Clone)]
#[structopt()]
pub struct LighthouseOpt {
    // bind is the address to bind the server to.
    #[structopt(
        long = "bind",
        default_value = "[::]:29510",
        help = "Address to bind the server to"
    )]
    pub bind: String,

    #[structopt(
        long = "join_timeout_ms",
        default_value = "60000",
        help = "How long to wait for heartbeating stragglers to join before issuing quorum"
    )]
    pub join_timeout_ms: u64,

    #[structopt(
        long = "min_replicas",
        help = "Minimum number of replicas to consider a quorum"
    )]
    pub min_replicas: u64,

    #[structopt(
        long = "quorum_tick_ms",
        default_value = "100",
        help = "How frequently to check for quorum when waiting for stragglers."
    )]
    pub quorum_tick_ms: u64,

    #[structopt(
        long = "heartbeat_timeout_ms",
        default_value = "5000",
        help = "How long to wait for a heartbeat before considering a replica dead."
    )]
    pub heartbeat_timeout_ms: u64,

    #[structopt(
        long = "failure_tick_ms",
        default_value = "1000",
        help = "How frequently to check for failures."
    )]
    pub failure_tick_ms: u64,

    #[structopt(
        long = "lighthouse_config",
        help = "Path to configuration file (JSON format)"
    )]
    pub lighthouse_config: Option<String>,
}

fn quorum_changed(a: &Vec<QuorumMember>, b: &Vec<QuorumMember>) -> bool {
    let a_ids: Vec<&String> = a.iter().map(|p| &p.replica_id).collect();
    let b_ids: Vec<&String> = b.iter().map(|p| &p.replica_id).collect();

    return a_ids != b_ids;
}

// Checks whether the quorum is valid, the new quorum and an explanation for the state.
fn quorum_compute(
    now: Instant,
    state: &State,
    opt: &LighthouseOpt,
) -> (Option<Vec<QuorumMember>>, String) {
    let heartbeats = &state.heartbeats;
    let healthy_replicas: HashSet<&String> = heartbeats
        .iter()
        .filter_map(|(replica_id, last_heartbeat)| {
            if now.duration_since(*last_heartbeat) < Duration::from_millis(opt.heartbeat_timeout_ms)
            {
                return Some(replica_id);
            }
            None
        })
        .collect();

    let healthy_participants: HashMap<&String, &QuorumMemberDetails> = state
        .participants
        .iter()
        .filter(|(replica_id, _details)| healthy_replicas.contains(replica_id))
        .collect();

    let mut candidate_participants: Vec<QuorumMember> = healthy_participants
        .values()
        .map(|details| details.member.clone())
        .collect();

    // Sort by replica ID to get a consistent ordering across runs.
    candidate_participants.sort_by_key(|p| p.replica_id.clone());

    let shrink_only = healthy_participants
        .iter()
        .any(|(_, details)| details.member.shrink_only);

    let metadata = format!(
        "[{}/{} participants healthy][{} heartbeating][shrink_only={}]",
        healthy_participants.len(),
        state.participants.len(),
        healthy_replicas.len(),
        shrink_only,
    );

    // Check if we can use the previous quorum.
    // TODO: do we still need this given we have heartbeats?
    if state.prev_quorum.is_some() {
        let prev_quorum = state.prev_quorum.as_ref().unwrap();

        let prev_replica_ids: HashSet<&String> = prev_quorum
            .participants
            .iter()
            .map(|p| &p.replica_id)
            .collect();

        if shrink_only {
            candidate_participants = candidate_participants
                .into_iter()
                .filter(|p| prev_replica_ids.contains(&p.replica_id))
                .collect();
        }

        // Fast quorum is when all previous participants are still in the quorum
        // and we have enough participants to form a quorum.
        let is_fast_quorum = prev_quorum
            .participants
            .iter()
            .all(|prev_member| healthy_participants.contains_key(&prev_member.replica_id));

        if is_fast_quorum {
            return (
                Some(candidate_participants),
                format!("Fast quorum found! {}", metadata),
            );
        }
    }

    // Minimum quorum size check.
    if healthy_participants.len() < opt.min_replicas as usize {
        return (
            None,
            format!(
                "New quorum not ready, only have {} participants, need min_replicas {} {}",
                healthy_participants.len(),
                opt.min_replicas,
                metadata
            ),
        );
    }

    // Avoid split brain by requiring at least half of the known alive workers.
    if healthy_participants.len() <= healthy_replicas.len() / 2 {
        return (
            None,
            format!(
                "New quorum not ready, only have {} participants, need at least half of {} healthy workers {}",
                healthy_participants.len(),
                healthy_replicas.len(),
                metadata
            ),
        );
    }

    let all_healthy_joined = healthy_participants.len() == healthy_replicas.len();

    // Quorum is valid at this point but lets wait for stragglers.
    let first_joined = healthy_participants
        .values()
        .map(|details| details.joined)
        .min()
        .unwrap_or(now);
    if !all_healthy_joined
        && now.duration_since(first_joined) < Duration::from_millis(opt.join_timeout_ms)
    {
        return (
            None,
            format!(
                "Valid quorum with {} participants, waiting for {} healthy but not participating stragglers due to join timeout {}",
                healthy_participants.len(),
                healthy_replicas.len() - healthy_participants.len(),
                metadata
            ),
        );
    }

    (
        Some(candidate_participants),
        format!("Valid quorum found {}", metadata),
    )
}

fn load_config(config_path: &Option<String>) -> serde_json::Map<String, serde_json::Value> {
    match config_path {
        Some(path) => {
            match fs::read_to_string(path) {
                Ok(content) => {
                    // Parse JSON into Map
                    match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
                        &content,
                    ) {
                        Ok(json_map) => {
                            info!("Successfully loaded config from {}", path);
                            json_map
                        }
                        Err(e) => {
                            warn!(
                                "Invalid JSON in config file {}: {}. Using empty config.",
                                path, e
                            );
                            serde_json::Map::new()
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to read config file {}: {}. Using empty config.",
                        path, e
                    );
                    serde_json::Map::new()
                }
            }
        }
        None => serde_json::Map::new(),
    }
}

impl Lighthouse {
    pub async fn new(opt: LighthouseOpt) -> Result<Arc<Self>> {
        let listener = tokio::net::TcpListener::bind(&opt.bind).await?;

        // Load configuration data
        let config_data = load_config(&opt.lighthouse_config);

        let (tx, _) = broadcast::channel(16);
        let (failure_tx, failure_rx) = broadcast::channel::<FailureNotification>(16);

        // Create a task to monitor the failure channel
        let mut failure_rx_cloned: broadcast::Receiver<FailureNotification> =
            failure_rx.resubscribe();
        tokio::spawn(async move {
            use tokio::time::{sleep, Duration};
            info!("Starting permanent failure channel subscriber");
            loop {
                match failure_rx_cloned.recv().await {
                    Ok(note) => {
                        info!(
                            "Healthy replicas received failure notification for {} with error message: {}",
                            note.replica_id,
                            note.error_message
                        );
                    }
                    Err(e) => {
                        error!("Healthy replicas error: {}", e);
                        // If the channel is closed, break the loop
                        if matches!(e, tokio::sync::broadcast::error::RecvError::Closed) {
                            break;
                        }
                    }
                }
                sleep(Duration::from_millis(100)).await; // Prevent thrashing if there are continuous errors
            }
            info!("Permanent failure channel subscriber exiting");
        });

        Ok(Arc::new(Self {
            state: Mutex::new(State {
                participants: HashMap::new(),
                quorum_channel: tx,
                prev_quorum: None,
                quorum_id: 0,
                heartbeats: HashMap::new(),
                failures: HashMap::new(),
                failure_channel: failure_tx,
                config_data: config_data,
            }),
            opt: opt,
            local_addr: listener.local_addr()?,
            listener: Mutex::new(Some(listener)),
            change_logger: ChangeLogger::new(),
        }))
    }

    fn _quorum_tick(self: Arc<Self>, state: &mut State) -> Result<()> {
        let (quorum_met, reason) = quorum_compute(Instant::now(), state, &self.opt);
        self.change_logger.log_if_changed(&reason);

        if quorum_met.is_some() {
            let participants = quorum_met.unwrap();

            let commit_failure_replica_ids: Vec<String> = participants
                .iter()
                .filter(|p| p.commit_failures > 0)
                .map(|p| p.replica_id.clone())
                .collect();

            // only increment quorum ID if something about the quorum
            // changed (members/addresses/etc)
            if state.prev_quorum.is_none()
                || quorum_changed(
                    &participants,
                    &state.prev_quorum.as_ref().unwrap().participants,
                )
            {
                state.quorum_id += 1;
                info!(
                    "Detected quorum change, bumping quorum_id to {}",
                    state.quorum_id
                );
            } else if commit_failure_replica_ids.len() > 0 {
                state.quorum_id += 1;
                info!(
                    "Detected commit failures in [{}], bumping quorum_id to {}",
                    commit_failure_replica_ids.join(", "),
                    state.quorum_id
                );
            }

            let quorum = Quorum {
                quorum_id: state.quorum_id,
                participants: participants,
                created: Some(SystemTime::now().into()),
            };

            info!("Quorum! {:?}", quorum);

            state.prev_quorum = Some(quorum.clone());
            state.participants.clear();
            match state.quorum_channel.send(quorum) {
                Ok(_) => (),
                Err(e) => error!("failed to send quorum {}", e),
            }
        }
        Ok(())
    }

    async fn _run_quorum(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.opt.quorum_tick_ms));
        loop {
            interval.tick().await; // Wait for the next tick
            let mut state = self.state.lock().await;
            self.clone()._quorum_tick(&mut state)?;
        }
    }

    pub fn address(&self) -> String {
        format!(
            "http://{}:{}",
            gethostname().into_string().unwrap(),
            self.local_addr.port()
        )
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info!("Lighthouse listening on: {}", self.address());

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        // Setup HTTP endpoints
        let app = Router::new()
            .route(
                "/",
                get(|| async { Html(IndexTemplate {}.render().unwrap()) }),
            )
            .route(
                "/status",
                get({
                    let self_clone = self.clone();
                    move || async { self_clone.get_status().await }
                }),
            )
            .route(
                "/config",
                get({
                    let self_clone = self.clone();
                    move || async { self_clone.get_config_page().await }
                })
                .put({
                    let self_clone = self.clone();
                    move |form_data: axum::extract::Form<ConfigUpdateForm>| async move {
                        self_clone.update_config(form_data).await
                    }
                }),
            )
            .route(
                "/replica/:replica_id/kill",
                post({
                    let self_clone = self.clone();
                    move |path| async { self_clone.kill(path).await }
                }),
            );

        // register the GRPC service
        let routes = Routes::from(app).add_service(LighthouseServiceServer::new(self));

        Server::builder()
            // allow non-GRPC connections
            .accept_http1(true)
            .add_routes(routes)
            .serve_with_incoming(incoming)
            .await
            .map_err(|e| e.into())
    }

    async fn _run_failure_tick(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.opt.failure_tick_ms));
        loop {
            interval.tick().await; // Wait for the next tick
            let mut state = self.state.lock().await;
            self.clone()._failure_tick(&mut state)?;
        }
    }

    fn _failure_tick(self: Arc<Self>, state: &mut State) -> Result<()> {
        let now = Instant::now();
        let timeout = Duration::from_millis(self.opt.heartbeat_timeout_ms);

        // Use a temporary list to collect replica IDs to remove from heartbeats
        // to avoid modifying the map while iterating over it.
        let mut failed_replica_ids_to_remove_from_heartbeats = Vec::new();
        let mut failure_detected = false;

        for (replica_id, last_heartbeat) in state.heartbeats.iter() {
            if now.duration_since(*last_heartbeat) > timeout {
                if !state.failures.contains_key(replica_id) {
                    info!(
                        "Replica {} timed out (last heartbeat: {:?}), sending failure notification.",
                        replica_id,
                        last_heartbeat
                    );
                    if let Err(e) = state.failure_channel.send(FailureNotification {
                        replica_id: replica_id.clone(),
                        error_message: "heartbeat timeout".to_string(),
                    }) {
                        error!(
                            "Failed to send failure notification for {}: {} (receiver count: {})",
                            replica_id,
                            e,
                            state.failure_channel.receiver_count()
                        );
                    } else {
                        failure_detected = true; // Set flag if notification sent successfully
                    }
                    // Record failure information
                    state.failures.insert(replica_id.clone(), now);
                    state.participants.remove(replica_id);
                    failed_replica_ids_to_remove_from_heartbeats.push(replica_id.clone());
                }
            } else {
                // If the participant sends heartbeat again, remove it from failures.
                if state.failures.remove(replica_id).is_some() {
                    info!("Replica {} recovered from failure.", replica_id);
                }
            }
        }

        // Remove failed replicas from heartbeats
        for replica_id in failed_replica_ids_to_remove_from_heartbeats {
            state.heartbeats.remove(&replica_id);
            info!(
                "Removed replica {} from heartbeats and participants due to timeout.",
                replica_id
            );
        }

        // If a new failure was detected and broadcasted, reset participants to restart quorum formation
        if failure_detected {
            info!("New failure detected, resetting all participants for quorum formation.");
            state.participants.clear();
        }

        Ok(())
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_quorum());

        set.spawn(self.clone()._run_grpc());

        set.spawn(self.clone()._run_failure_tick());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }

    async fn get_status(self: Arc<Self>) -> Html<String> {
        let template = {
            let state = self.state.lock().await;

            let (_, quorum_status) = quorum_compute(Instant::now(), &state, &self.opt);

            let max_step = if let Some(quorum) = &state.prev_quorum {
                quorum
                    .participants
                    .iter()
                    .map(|p| p.step)
                    .max()
                    .unwrap_or(-1)
            } else {
                -1
            };

            let num_participants = if let Some(quorum) = &state.prev_quorum {
                quorum.participants.len() as i64
            } else {
                -1
            };

            StatusTemplate {
                quorum_id: state.quorum_id,
                num_participants: num_participants,
                prev_quorum: state.prev_quorum.clone(),
                quorum_status: quorum_status,
                max_step: max_step,

                heartbeats: state.heartbeats.clone(),
                old_age_threshold: Instant::now()
                    .checked_sub(Duration::from_millis(self.opt.heartbeat_timeout_ms))
                    .unwrap_or(Instant::now()),
            }
        };
        Html(template.render().unwrap())
    }

    async fn kill(self: Arc<Self>, Path(replica_id): Path<String>) -> Result<(), AppError> {
        let addr = 'addr: {
            let state = self.state.lock().await;

            if state.prev_quorum.is_none() {
                return Err(AppError(anyhow!("failed to find replica")));
            }

            for member in state.prev_quorum.clone().unwrap().participants {
                if member.replica_id == replica_id {
                    break 'addr member.address;
                }
            }

            return Err(AppError(anyhow!("failed to find replica")));
        };

        let mut client = manager_client_new(addr, Duration::from_secs(10)).await?;

        let request = tonic::Request::new(KillRequest {
            msg: "killed from dashboard".to_string(),
        });
        let _resp = client.kill(request).await?;

        Ok(())
    }

    async fn get_config_page(self: Arc<Self>) -> Html<String> {
        self.get_config_page_with_message("".to_string()).await
    }

    async fn get_config_page_with_message(
        self: Arc<Self>,
        success_message: String,
    ) -> Html<String> {
        let config_data = {
            let state = self.state.lock().await;
            // Serialize Map to JSON string for the web interface
            match serde_json::to_string_pretty(&state.config_data) {
                Ok(json_str) => json_str,
                Err(_) => "{}".to_string(),
            }
        };

        let timestamp = chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string();

        let template = ConfigTemplate {
            config_data: config_data,
            timestamp: timestamp,
            success_message: if success_message.is_empty() {
                None
            } else {
                Some(success_message)
            },
        };
        Html(template.render().unwrap())
    }

    async fn update_config(
        self: Arc<Self>,
        axum::extract::Form(form_data): axum::extract::Form<ConfigUpdateForm>,
    ) -> Result<Html<String>, AppError> {
        let new_config_json = form_data.config;

        info!("Update config called with: {}", new_config_json);

        // Parse and validate the JSON into a Map
        let new_config_map = match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &new_config_json,
        ) {
            Ok(json_map) => json_map,
            Err(e) => {
                warn!("Invalid JSON provided via web interface: {}", e);
                return Err(AppError(anyhow!("Invalid JSON: {}", e)));
            }
        };

        // Update the config in the lighthouse state
        {
            let mut state = self.state.lock().await;
            state.config_data = new_config_map.clone();
        }

        // Log the updated configuration content
        match serde_json::to_string_pretty(&new_config_map) {
            Ok(pretty_json) => {
                info!(
                    "Config updated successfully via web interface. New configuration:\n{}",
                    pretty_json
                );
            }
            Err(_) => {
                info!(
                    "Config updated successfully via web interface. New configuration: {:?}",
                    new_config_map
                );
            }
        }

        // Return the updated config page with success message
        Ok(self
            .get_config_page_with_message("Configuration updated successfully!".to_string())
            .await)
    }

    pub async fn inject_failure(self: Arc<Self>, replica_id: String) -> Result<()> {
        let state = self.state.lock().await;
        state
            .failure_channel
            .send(FailureNotification {
                replica_id,
                error_message: "injected failure".to_string(),
            })
            .map_err(|e| anyhow!("Failed to send failure notification: {}", e))?;
        Ok(())
    }
}

#[tonic::async_trait]
impl LighthouseService for Arc<Lighthouse> {
    async fn quorum(
        &self,
        request: Request<LighthouseQuorumRequest>,
    ) -> Result<Response<LighthouseQuorumResponse>, Status> {
        let req = request.into_inner();
        let requester = req
            .requester
            .ok_or_else(|| return Status::invalid_argument("missing requester"))?;

        info!(
            "Received quorum request for replica {}",
            &requester.replica_id
        );

        let mut rx = {
            let mut state = self.state.lock().await;

            // implicit heartbeat
            state
                .heartbeats
                .insert(requester.replica_id.clone(), Instant::now());

            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester.clone(),
                },
            );
            let rx = state.quorum_channel.subscribe();

            // proactively run quorum tick
            self.clone()
                ._quorum_tick(&mut state)
                .map_err(|e| Status::from_error(e.into()))?;

            rx
        };

        let quorum = loop {
            let current_quorum = rx.recv().await.map_err(|e| Status::from_error(e.into()))?;

            if current_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == requester.replica_id)
            {
                break current_quorum;
            }

            // Only continue the loop if the replica is not in the quorum
            let mut state = self.state.lock().await;
            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester.clone(),
                },
            );
            info!("Replica {} not in quorum, retrying", &requester.replica_id);
        };

        let reply = LighthouseQuorumResponse {
            quorum: Some(quorum),
        };

        Ok(Response::new(reply))
    }

    async fn heartbeat(
        &self,
        request: Request<LighthouseHeartbeatRequest>,
    ) -> Result<Response<LighthouseHeartbeatResponse>, Status> {
        let replica_id = request.into_inner().replica_id;

        {
            let mut state = self.state.lock().await;
            state.heartbeats.insert(replica_id, Instant::now());
        }

        let reply = LighthouseHeartbeatResponse {};
        Ok(Response::new(reply))
    }

    type SubscribeFailuresStream =
        Pin<Box<dyn Stream<Item = Result<FailureNotification, Status>> + Send + 'static>>;

    async fn subscribe_failures(
        &self,
        _req: Request<SubscribeFailuresRequest>,
    ) -> Result<Response<Self::SubscribeFailuresStream>, Status> {
        // clone a receiver
        let rx = {
            let state = self.state.lock().await;
            let receiver_count = state.failure_channel.receiver_count();
            info!(
                "subscribe_failures: Creating new subscriber (current count: {})",
                receiver_count
            );
            state.failure_channel.subscribe()
        };

        // Wrap the receiver; map its *internal* error into `tonic::Status`
        let stream = BroadcastStream::new(rx).filter_map(|res| match res {
            Ok(note) => Some(Ok(note)),
            Err(TokioStreamBroadcastStreamRecvError::Lagged(n)) => Some(Err(
                Status::resource_exhausted(format!("client lagged {n} messages")),
            )),
        });

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_config(
        &self,
        _request: Request<LighthouseConfigRequest>,
    ) -> Result<Response<LighthouseConfigResponse>, Status> {
        let config_data = {
            let state = self.state.lock().await;
            // Convert serde_json::Map to HashMap<String, String> for protobuf
            state
                .config_data
                .iter()
                .map(|(k, v)| {
                    let value_str = match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    };
                    (k.clone(), value_str)
                })
                .collect()
        };

        let reply = LighthouseConfigResponse { config_data };
        Ok(Response::new(reply))
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

#[derive(Template)]
#[template(path = "status.html")]
struct StatusTemplate {
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
    quorum_status: String,
    num_participants: i64,
    max_step: i64,
    heartbeats: HashMap<String, Instant>,

    // visualization thresholds
    old_age_threshold: Instant,
}

#[derive(Template)]
#[template(path = "config.html")]
struct ConfigTemplate {
    config_data: String,
    timestamp: String,
    success_message: Option<String>,
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[derive(Deserialize)]
struct ConfigUpdateForm {
    config: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Sub;
    use tokio::sync::broadcast::error::RecvError as TokioBroadcastRecvError;
    use tokio::time::timeout as tokio_timeout;

    use tonic::transport::Channel;

    use crate::net::connect;
    use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;

    async fn lighthouse_client_new(addr: String) -> Result<LighthouseServiceClient<Channel>> {
        let conn = connect(addr, Duration::from_secs(10)).await?;
        Ok(LighthouseServiceClient::new(conn))
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let mut state = State {
            quorum_channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            failures: HashMap::new(),
            failure_channel: broadcast::channel(16).0,
            config_data: serde_json::Map::new(),
        };

        let now = Instant::now();

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("New quorum not ready, only have 0 participants, need min_replicas 1 [0/0 participants healthy]"), "{}", reason);

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        // all healthy workers participating
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        // add healthy worker but not participating
        state.heartbeats.insert("c".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("join timeout"), "{}", reason);

        // increase elapsed time to pass join timeout
        state.participants.get_mut("a").unwrap().joined =
            now.sub(Duration::from_secs(10 * 60 * 60));
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_heartbeats() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let mut state = State {
            quorum_channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            failures: HashMap::new(),
            failure_channel: broadcast::channel(16).0,
            config_data: serde_json::Map::new(),
        };

        let now = Instant::now();

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        assert!(
            reason.contains("[1/1 participants healthy][1 heartbeating]"),
            "{}",
            reason
        );

        // expired heartbeat
        state
            .heartbeats
            .insert("a".to_string(), now.sub(Duration::from_secs(10)));

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(
            reason.contains("[0/1 participants healthy][0 heartbeating]"),
            "{}",
            reason
        );

        // 1 healthy, 1 expired
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let mut state = State {
            quorum_channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            failures: HashMap::new(),
            failure_channel: broadcast::channel(16).0,
            config_data: serde_json::Map::new(),
        };

        let now = Instant::now();

        assert!(!quorum_compute(now, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        // Not proceeding since one worker is alive but not participating
        state.heartbeats.insert("b".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("need at least half"), "{}", reason);

        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![QuorumMember {
                replica_id: "a".to_string(),
                address: "".to_string(),
                store_address: "".to_string(),
                step: 1,
                world_size: 1,
                shrink_only: false,
                data: String::new(),
                commit_failures: 0,
            }],
            created: Some(SystemTime::now().into()),
        });

        assert!(quorum_compute(now, &state, &opt).0.is_some());

        // test expanding quorum w/ fast quorum
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_shrink_only() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let mut state = State {
            quorum_channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            failures: HashMap::new(),
            failure_channel: broadcast::channel(16).0,
            config_data: serde_json::Map::new(),
        };

        let now = Instant::now();

        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: Some(SystemTime::now().into()),
        });

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: true,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        // insert particpant that was not in prev quorum
        state.participants.insert(
            "c".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "c".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: true,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("c".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        assert!(reason.contains("[shrink_only=true]",), "{}", reason);

        let quorum = quorum_met.unwrap();
        assert!(quorum.len() == 1);
        assert!(quorum[0].replica_id == "a");

        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_e2e() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };
        let lighthouse = Lighthouse::new(opt).await?;

        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await.unwrap();

        {
            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: "foo".to_string(),
            });

            let _response = client.heartbeat(request).await.unwrap();
        }

        {
            let request = tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(QuorumMember {
                    replica_id: "foo".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 10,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                }),
            });

            let response = client.quorum(request).await.unwrap();
            let quorum = response.into_inner().quorum.unwrap();
            assert_eq!(quorum.participants.len(), 1);
        }

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_split_brain() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let mut state = State {
            quorum_channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            failures: HashMap::new(),
            failure_channel: broadcast::channel(16).0,
            config_data: serde_json::Map::new(),
        };

        let now = Instant::now();

        assert!(!quorum_compute(now, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        // Not proceeding since one worker is alive but not participating
        state.heartbeats.insert("b".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("New quorum not ready, only have 1 participants, need at least half of 2 healthy workers [1/1 participants healthy][2 heartbeating]"), "{}", reason);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_changed() {
        let a = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];
        let b = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "changed".to_string(),
            store_address: "changed".to_string(),
            step: 1000,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];

        // replica_id is the same
        assert!(!quorum_changed(&a, &b));

        let c = vec![QuorumMember {
            replica_id: "2".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];
        // replica_id changed
        assert!(quorum_changed(&a, &c));
    }

    // Helper to create a default QuorumMember for tests
    fn test_quorum_member(replica_id: &str) -> QuorumMember {
        QuorumMember {
            replica_id: replica_id.to_string(),
            address: format!("addr_{}", replica_id),
            store_address: format!("store_{}", replica_id),
            step: 1,
            world_size: 2, // Assuming 2 for this test context
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }
    }

    /// Test that `_failure_tick` correctly identifies timed-out replicas,
    /// broadcasts a failure notification exactly once per failure, and
    /// cleans up the replica from `heartbeats` and `participants` while
    /// adding it to `failures`. Subsequent ticks should not re-notify
    /// or change the state for an already failed replica.
    #[tokio::test]
    async fn test_failure_tick_single_notification_and_cleanup() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,        // Not relevant for this test
            quorum_tick_ms: 10,        // Not directly relevant but keep it small
            heartbeat_timeout_ms: 100, // Reasonably short for testing
            failure_tick_ms: 50,       // How often _failure_tick would be called
            lighthouse_config: None,
        };
        let lighthouse = Lighthouse::new(opt.clone()).await?;

        let mut failure_rx = {
            let state_guard = lighthouse.state.lock().await;
            state_guard.failure_channel.subscribe()
        };

        let replica_id_failing = "failing_one";

        let now = Instant::now();
        // Ensure expired_time is definitively older than heartbeat_timeout_ms
        let expired_time = now - Duration::from_millis(opt.heartbeat_timeout_ms * 2);

        // Setup initial state: one about to fail
        {
            let mut state_guard = lighthouse.state.lock().await;
            let state = &mut *state_guard;

            // Failing replica
            state.participants.insert(
                replica_id_failing.to_string(),
                QuorumMemberDetails {
                    joined: now, // Joined time doesn't prevent failure due to heartbeat
                    member: test_quorum_member(replica_id_failing),
                },
            );
            state
                .heartbeats
                .insert(replica_id_failing.to_string(), expired_time);
        }

        // --- First call to _failure_tick ---
        // This call should detect the failure, send a notification, and update state.
        {
            let mut state_guard = lighthouse.state.lock().await;
            lighthouse.clone()._failure_tick(&mut *state_guard)?;
        }

        // Assertions after first tick
        // 1. Check notification for failing_replica
        match tokio_timeout(
            Duration::from_millis(opt.failure_tick_ms * 2),
            failure_rx.recv(),
        )
        .await
        {
            Ok(Ok(notification)) => {
                assert_eq!(
                    notification.replica_id, replica_id_failing,
                    "Notification should be for the failing replica"
                );
            }
            Ok(Err(TokioBroadcastRecvError::Lagged(n))) => {
                panic!(
                    "Broadcast channel lagged by {} messages, missed the failure notification",
                    n
                );
            }
            Ok(Err(TokioBroadcastRecvError::Closed)) => {
                panic!("Broadcast channel closed unexpectedly after first tick");
            }
            Err(_) => panic!(
                "Did not receive failure notification for {} in time",
                replica_id_failing
            ),
        }

        // 2. Verify state changes
        {
            let state_guard = lighthouse.state.lock().await;
            let state = &*state_guard;

            // Failing replica assertions
            assert!(
                state.failures.contains_key(replica_id_failing),
                "{} should be in failures map",
                replica_id_failing
            );
            assert!(
                !state.heartbeats.contains_key(replica_id_failing),
                "{} should be removed from heartbeats",
                replica_id_failing
            );
            assert!(
                !state.participants.contains_key(replica_id_failing),
                "{} should be removed from participants",
                replica_id_failing
            );
        }

        // --- Second call to _failure_tick ---
        // This call should *not* detect a *new* failure for the same replica
        // and should not send another notification.
        {
            let mut state_guard = lighthouse.state.lock().await;
            lighthouse.clone()._failure_tick(&mut *state_guard)?;
        }

        // Assertions after second tick
        // 1. No new notification for failing_replica
        match tokio_timeout(
            Duration::from_millis(opt.failure_tick_ms * 2),
            failure_rx.recv(),
        )
        .await
        {
            Ok(Ok(notification)) => {
                panic!(
                    "Received unexpected second failure notification for {}",
                    notification.replica_id
                );
            }
            Ok(Err(TokioBroadcastRecvError::Lagged(n))) => {
                // This might happen if the test environment is slow and ticks are processed faster than receives.
                // For this specific assertion (no *new* message), lagging is an acceptable outcome.
                info!("Broadcast channel lagged by {} messages on second check, implies no new distinct message.", n);
            }
            Ok(Err(TokioBroadcastRecvError::Closed)) => {
                // Channel might close if sender is dropped, implies no new message.
                info!("Broadcast channel closed on second check, implies no new distinct message.");
            }
            Err(_) => {
                // Expected: Timeout, meaning no new message was received for failing_replica.
            }
        }

        // 2. Verify state remains consistent for failing_replica
        {
            let state_guard = lighthouse.state.lock().await;
            let state = &*state_guard;

            assert!(
                state.failures.contains_key(replica_id_failing),
                "{} should remain in failures map",
                replica_id_failing
            );
            assert!(
                !state.heartbeats.contains_key(replica_id_failing),
                "{} should remain removed from heartbeats",
                replica_id_failing
            );
            assert!(
                !state.participants.contains_key(replica_id_failing),
                "{} should remain removed from participants",
                replica_id_failing
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_join_during_shrink() -> Result<()> {
        fn create_member(id: &str, addr_num: &str, step: i64, shrink_only: bool) -> QuorumMember {
            QuorumMember {
                replica_id: id.to_string(),
                address: format!("addr{}", addr_num),
                store_address: format!("store{}", addr_num),
                step,
                world_size: 1,
                shrink_only,
                data: String::new(),
                commit_failures: 0,
            }
        }

        fn create_request(member: &QuorumMember) -> tonic::Request<LighthouseQuorumRequest> {
            tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(member.clone()),
            })
        }

        let opt = LighthouseOpt {
            min_replicas: 2,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        // Start the lighthouse service
        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // Create client to interact with lighthouse
        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        // 1. First quorum
        let mut first_request = create_request(&create_member("replica0", "1", 1, false));
        let mut second_request = create_request(&create_member("replica1", "2", 1, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let first_response = client.quorum(second_request).await?;
        let first_quorum = first_response.into_inner().quorum.unwrap();
        assert_eq!(first_quorum.participants.len(), 2);
        assert_eq!(first_quorum.participants[0].replica_id, "replica0");
        assert_eq!(first_quorum.participants[1].replica_id, "replica1");
        assert_eq!(first_quorum.participants[1].step, 1);

        // 2. Quorum without joiner
        let joining_request = create_request(&create_member("joiner", "3", 1, false));
        let joining_task = tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(joining_request).await }
        });

        // Try to shrink only
        first_request = create_request(&create_member("replica0", "1", 2, true));
        second_request = create_request(&create_member("replica1", "2", 2, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let second_response = client.quorum(second_request).await?;
        let second_quorum = second_response.into_inner().quorum.unwrap();
        assert!(second_quorum
            .participants
            .iter()
            .all(|p| p.replica_id != "joiner"));
        assert_eq!(second_quorum.participants.len(), 2);
        assert_eq!(second_quorum.participants[0].replica_id, "replica0");
        assert_eq!(second_quorum.participants[1].replica_id, "replica1");
        assert_eq!(second_quorum.participants[1].step, 2);

        // 3. Quorum with joiner
        first_request = create_request(&create_member("replica0", "1", 3, false));
        second_request = create_request(&create_member("replica1", "2", 3, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let third_response = client.quorum(second_request).await?;
        let third_quorum = third_response.into_inner().quorum.unwrap();
        assert!(third_quorum
            .participants
            .iter()
            .any(|p| p.replica_id == "joiner"));
        assert_eq!(third_quorum.participants.len(), 3);
        assert_eq!(third_quorum.participants[2].step, 3);

        let join_result = joining_task.await?;
        let join_quorum = join_result.unwrap().into_inner().quorum.unwrap();
        assert!(join_quorum
            .participants
            .iter()
            .any(|p| p.replica_id == "joiner"));
        assert_eq!(join_quorum.participants.len(), 3);
        assert_eq!(join_quorum.participants[2].step, 3);

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_commit_failures() -> Result<()> {
        fn create_member(id: &str, commit_failures: i64) -> QuorumMember {
            QuorumMember {
                replica_id: id.to_string(),
                address: format!("addr{}", id),
                store_address: format!("store{}", id),
                step: 10,
                world_size: 1,
                shrink_only: false,
                data: String::new(),
                commit_failures,
            }
        }

        fn create_request(member: &QuorumMember) -> tonic::Request<LighthouseQuorumRequest> {
            tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(member.clone()),
            })
        }

        let opt = LighthouseOpt {
            min_replicas: 2,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        // Start the lighthouse service
        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // Create client to interact with lighthouse
        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        // First two quorums should be stable
        for _i in 0..2 {
            let first_request = create_request(&create_member("replica0", 0));
            let second_request = create_request(&create_member("replica1", 0));

            tokio::spawn({
                let mut client = client.clone();
                async move { client.quorum(first_request).await }
            });
            let first_response = client.quorum(second_request).await?;
            let first_quorum = first_response.into_inner().quorum.unwrap();
            assert_eq!(first_quorum.quorum_id, 1);
            assert_eq!(first_quorum.participants.len(), 2);
            assert_eq!(first_quorum.participants[0].commit_failures, 0);
            assert_eq!(first_quorum.participants[1].commit_failures, 0);
        }

        // commit_failures should increment quorum_id
        let first_request = create_request(&create_member("replica0", 0));
        let second_request = create_request(&create_member("replica1", 2));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let first_response = client.quorum(second_request).await?;
        let first_quorum = first_response.into_inner().quorum.unwrap();
        assert_eq!(first_quorum.quorum_id, 2);
        assert_eq!(first_quorum.participants.len(), 2);
        assert_eq!(first_quorum.participants[0].commit_failures, 0);
        assert_eq!(first_quorum.participants[1].commit_failures, 2);

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_subscribe_failures_basic() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await?;
        let request = tonic::Request::new(SubscribeFailuresRequest {});
        client.subscribe_failures(request).await?;

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_subscribe_failures_delivers_notifications() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };
        let lighthouse = Lighthouse::new(opt).await?;
        let mut client = lighthouse_client_new(lighthouse.address()).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // 1. Subscribe with a deadline
        let mut req = tonic::Request::new(SubscribeFailuresRequest {});
        req.set_timeout(Duration::from_secs(5));
        let mut stream = client.subscribe_failures(req).await?.into_inner();

        // 2. Trigger a failure notification
        {
            let state = lighthouse.state.lock().await;
            state
                .failure_channel
                .send(FailureNotification {
                    replica_id: "replica_id_X".into(),
                    error_message: "injected failure".to_string(),
                })
                .unwrap();
        }

        // 3. Ensure we receive it
        match stream.next().await {
            Some(Ok(note)) => {
                assert_eq!(note.replica_id, "replica_id_X");
                assert_eq!(note.error_message, "injected failure");
            }
            other => panic!("Expected notification, got {:?}", other),
        }

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_config_broadcasting() -> Result<()> {
        // Create a test config file
        let test_config =
            r#"{"learning_rate": "0.001", "batch_size": "32", "model_type": "transformer"}"#;
        std::fs::write("test_config_temp.json", test_config).unwrap();

        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60000,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: Some("test_config_temp.json".to_string()),
        };

        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        let request = tonic::Request::new(LighthouseConfigRequest {});
        let response = client.get_config(request).await?;
        let config_data = response.into_inner().config_data;

        // Verify the config was loaded and parsed correctly
        assert_eq!(config_data.get("learning_rate"), Some(&"0.001".to_string()));
        assert_eq!(config_data.get("batch_size"), Some(&"32".to_string()));
        assert_eq!(
            config_data.get("model_type"),
            Some(&"transformer".to_string())
        );
        assert_eq!(config_data.len(), 3);

        lighthouse_task.abort();

        // Clean up test file
        std::fs::remove_file("test_config_temp.json").unwrap();

        Ok(())
    }

    #[tokio::test]
    async fn test_config_broadcasting_no_config() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60000,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: None,
        };

        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        let request = tonic::Request::new(LighthouseConfigRequest {});
        let response = client.get_config(request).await?;
        let config_data = response.into_inner().config_data;

        // When no config is provided, should return empty map
        assert_eq!(config_data.len(), 0);

        lighthouse_task.abort();

        Ok(())
    }

    #[tokio::test]
    async fn test_config_broadcasting_invalid_json() -> Result<()> {
        // Create an invalid JSON file
        let invalid_json = r#"{"learning_rate": "0.001", "batch_size": 32 // invalid comment"#;
        std::fs::write("test_invalid_config.json", invalid_json).unwrap();

        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60000,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
            failure_tick_ms: 1000,
            lighthouse_config: Some("test_invalid_config.json".to_string()),
        };

        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        let request = tonic::Request::new(LighthouseConfigRequest {});
        let response = client.get_config(request).await?;
        let config_data = response.into_inner().config_data;

        // When invalid JSON is provided, should return empty map
        assert_eq!(config_data.len(), 0);

        lighthouse_task.abort();

        // Clean up test file
        std::fs::remove_file("test_invalid_config.json").unwrap();

        Ok(())
    }
}
