// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod lighthouse;
pub mod manager;
mod net;
mod retry;
mod timeout;

use anyhow::Result;
use atty::Stream;
use core::time::Duration;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use std::env;
use std::sync::Arc;
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tonic::transport::Channel;
use tonic::Status;

use chrono::Local;
use fern::colors::{Color, ColoredLevelConfig};
use log::LevelFilter;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::{CheckpointMetadataRequest, ManagerQuorumRequest, ShouldCommitRequest};
use pyo3::prelude::*;

/// ManagerServer is a GRPC server for the manager service.
/// There should be one manager server per replica group (typically running on
/// the rank 0 host). The individual ranks within a replica group should use
/// ManagerClient to communicate with the manager server and participate in
/// quorum operations.
///
/// Args:
///     replica_id (str): The ID of the replica group.
///     lighthouse_addr (str): The HTTP address of the lighthouse server.
///     hostname (str): The hostname of the manager server.
///     bind (str): The HTTP address to bind the server to.
///     store_addr (str): The HTTP address of the store server.
///     world_size (int): The world size of the replica group.
///     heartbeat_interval (timedelta): The interval at which heartbeats are sent.
///     connect_timeout (timedelta): The timeout for connecting to the lighthouse server.
#[pyclass]
struct ManagerServer {
    handle: JoinHandle<Result<()>>,
    manager: Arc<manager::Manager>,
    _runtime: Runtime,
}

#[pymethods]
impl ManagerServer {
    #[new]
    fn new(
        py: Python<'_>,
        replica_id: String,
        lighthouse_addr: String,
        hostname: String,
        bind: String,
        store_addr: String,
        world_size: u64,
        heartbeat_interval: Duration,
        connect_timeout: Duration,
    ) -> PyResult<Self> {
        py.allow_threads(move || {
            let runtime = Runtime::new()?;
            let manager = runtime
                .block_on(manager::Manager::new(
                    replica_id,
                    lighthouse_addr,
                    hostname,
                    bind,
                    store_addr,
                    world_size,
                    heartbeat_interval,
                    connect_timeout,
                ))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let handle = runtime.spawn(manager.clone().run());
            Ok(Self {
                handle: handle,
                manager: manager,
                _runtime: runtime,
            })
        })
    }

    /// address returns the address of the manager server.
    ///
    /// Returns:
    ///   str: The address of the manager server.
    fn address(&self) -> PyResult<String> {
        Ok(self.manager.address().to_string())
    }

    /// shutdown shuts down the manager server.
    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(move || {
            self.handle.abort();
        })
    }
}

/// ManagerClient is a GRPC client to the manager service.
///
/// It is used by the trainer to communicate with the ManagerServer.
///
/// Args:
///     addr (str): The HTTP address of the manager server.
///     connect_timeout (timedelta): The timeout for connecting to the manager server.
#[pyclass]
struct ManagerClient {
    runtime: Runtime,
    client: ManagerServiceClient<Channel>,
}

#[pymethods]
impl ManagerClient {
    #[new]
    fn new(py: Python<'_>, addr: String, connect_timeout: Duration) -> PyResult<Self> {
        py.allow_threads(move || {
            let runtime = Runtime::new()?;
            let client = runtime
                .block_on(manager::manager_client_new(addr, connect_timeout))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(Self {
                runtime: runtime,
                client: client,
            })
        })
    }

    fn _quorum(
        &self,
        py: Python<'_>,
        rank: i64,
        step: i64,
        checkpoint_metadata: String,
        shrink_only: bool,
        timeout: Duration,
    ) -> Result<QuorumResult, StatusError> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(ManagerQuorumRequest {
                rank: rank,
                step: step,
                checkpoint_metadata: checkpoint_metadata,
                shrink_only: shrink_only,
            });

            // This timeout is processed on the server side so we also enable
            // keep alives to detect server health.
            request.set_timeout(timeout);

            let response = self.runtime.block_on(self.client.clone().quorum(request))?;
            let resp = response.into_inner();
            Ok(QuorumResult {
                quorum_id: resp.quorum_id,
                replica_rank: resp.replica_rank,
                replica_world_size: resp.replica_world_size,
                recover_src_manager_address: resp.recover_src_manager_address,
                recover_src_rank: resp.recover_src_rank,
                recover_dst_ranks: resp.recover_dst_ranks,
                store_address: resp.store_address,
                max_step: resp.max_step,
                max_rank: resp.max_rank,
                max_world_size: resp.max_world_size,
                heal: resp.heal,
            })
        })
    }

    fn _checkpoint_metadata(
        &self,
        py: Python<'_>,
        rank: i64,
        timeout: Duration,
    ) -> Result<String, StatusError> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(CheckpointMetadataRequest { rank: rank });

            // This timeout is processed on the server side so we also enable
            // keep alives to detect server health.
            request.set_timeout(timeout);

            let response = self
                .runtime
                .block_on(self.client.clone().checkpoint_metadata(request))?;
            let resp = response.into_inner();
            Ok(resp.checkpoint_metadata)
        })
    }

    /// should_commit makes a request to the manager to determine if the trainer
    /// should commit the current step. This waits until all ranks check in at
    /// the specified step and will return false if any worker passes
    /// ``should_commit=False``.
    ///
    /// Args:
    ///     rank (int): The rank of the trainer.
    ///     step (int): The step of the trainer.
    ///     should_commit (bool): Whether the trainer should commit the current step.
    ///     timeout (timedelta): The timeout for the request. If the request
    ///         times out a TimeoutError is raised.
    ///
    /// Returns:
    ///    bool: Whether the trainer should commit the current step.
    fn should_commit(
        &self,
        py: Python<'_>,
        rank: i64,
        step: i64,
        should_commit: bool,
        timeout: Duration,
    ) -> Result<bool, StatusError> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(ShouldCommitRequest {
                rank: rank,
                step: step,
                should_commit: should_commit,
            });

            // This notifies the server about the timeout but doesn't affect the
            // endpoint timeout which we set on client creation.
            request.set_timeout(timeout);

            let response = self
                .runtime
                .block_on(self.client.clone().should_commit(request))?;
            let resp = response.into_inner();
            Ok(resp.should_commit)
        })
    }
}

#[pyclass(get_all, set_all)]
struct QuorumResult {
    quorum_id: i64,
    replica_rank: i64,
    replica_world_size: i64,
    recover_src_manager_address: String,
    recover_src_rank: Option<i64>,
    recover_dst_ranks: Vec<i64>,
    store_address: String,
    max_step: i64,
    max_rank: Option<i64>,
    max_world_size: i64,
    heal: bool,
}

#[pymethods]
impl QuorumResult {
    #[new]
    fn new() -> Self {
        Self {
            quorum_id: 0,
            replica_rank: 0,
            replica_world_size: 1,
            recover_src_manager_address: "".to_string(),
            recover_src_rank: None,
            recover_dst_ranks: Vec::new(),
            store_address: "".to_string(),
            max_step: 0,
            max_rank: None,
            max_world_size: 1,
            heal: false,
        }
    }
}

fn reset_python_signals(py: Python<'_>) -> PyResult<()> {
    // clear python signal handlers
    // signal.signal(signal.SIGINT, signal.SIG_DFL)
    let signal = py.import_bound("signal")?;
    let set_signal = signal.getattr("signal")?;
    let args = (signal.getattr("SIGINT")?, signal.getattr("SIG_DFL")?);
    set_signal.call1(args)?;

    Ok(())
}

#[pyfunction]
fn lighthouse_main(py: Python<'_>) -> PyResult<()> {
    reset_python_signals(py)?;

    let mut args = env::args();
    args.next(); // discard binary arg
    let opt = lighthouse::LighthouseOpt::from_iter(args);
    let rt = Runtime::new()?;
    rt.block_on(lighthouse_main_async(opt))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

async fn lighthouse_main_async(opt: lighthouse::LighthouseOpt) -> Result<()> {
    let lighthouse = lighthouse::Lighthouse::new(opt).await?;

    lighthouse.run().await?;

    Ok(())
}

/// LighthouseServer is a GRPC server for the lighthouse service.
///
/// It is used to coordinate the ManagerServer for each replica group.
///
/// This entrypoint is primarily for testing and debugging purposes. The
/// ``torchft_lighthouse`` command is recommended for most use cases.
///
/// Args:
///     bind (str): The HTTP address to bind the server to.
///     min_replicas (int): The minimum number of replicas required to form a quorum.
///     join_timeout_ms (int): The timeout for joining the quorum.
///     quorum_tick_ms (int): The interval at which the quorum is checked.
///     heartbeat_timeout_ms (int): The timeout for heartbeats.
#[pyclass]
struct LighthouseServer {
    lighthouse: Arc<lighthouse::Lighthouse>,
    handle: JoinHandle<Result<()>>,
    _runtime: Runtime,
}

#[pymethods]
impl LighthouseServer {
    #[pyo3(signature = (bind, min_replicas, join_timeout_ms=None, quorum_tick_ms=None, heartbeat_timeout_ms=None))]
    #[new]
    fn new(
        py: Python<'_>,
        bind: String,
        min_replicas: u64,
        join_timeout_ms: Option<u64>,
        quorum_tick_ms: Option<u64>,
        heartbeat_timeout_ms: Option<u64>,
    ) -> PyResult<Self> {
        let join_timeout_ms = join_timeout_ms.unwrap_or(100);
        let quorum_tick_ms = quorum_tick_ms.unwrap_or(100);
        let heartbeat_timeout_ms = heartbeat_timeout_ms.unwrap_or(5000);

        py.allow_threads(move || {
            let rt = Runtime::new()?;

            let lighthouse = rt
                .block_on(lighthouse::Lighthouse::new(lighthouse::LighthouseOpt {
                    bind: bind,
                    min_replicas: min_replicas,
                    join_timeout_ms: join_timeout_ms,
                    quorum_tick_ms: quorum_tick_ms,
                    heartbeat_timeout_ms: heartbeat_timeout_ms,
                }))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(Self {
                handle: rt.spawn(lighthouse.clone().run()),
                lighthouse: lighthouse,
                _runtime: rt,
            })
        })
    }

    /// address returns the address of the lighthouse server.
    ///
    /// Returns:
    ///    str: The address of the lighthouse server.
    fn address(&self) -> PyResult<String> {
        Ok(self.lighthouse.address().to_string())
    }

    /// shutdown shuts down the lighthouse server.
    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(move || {
            self.handle.abort();
        })
    }
}

struct StatusError(Status);

impl From<StatusError> for PyErr {
    fn from(error: StatusError) -> Self {
        let code = error.0.code();
        match code {
            tonic::Code::Cancelled | tonic::Code::DeadlineExceeded => {
                PyTimeoutError::new_err(error.0.to_string())
            }
            _ => PyRuntimeError::new_err(error.0.to_string()),
        }
    }
}

impl From<Status> for StatusError {
    fn from(other: Status) -> Self {
        Self(other)
    }
}

fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    // Check if stderr is a terminal
    let is_terminal = atty::is(Stream::Stderr);
    let colors = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .info(Color::Green)
        .debug(Color::Blue)
        .trace(Color::Magenta);
    let level_filter = match env::var("RUST_LOG").as_deref() {
        Ok("error") => LevelFilter::Error,
        Ok("warn") => LevelFilter::Warn,
        Ok("info") => LevelFilter::Info,
        Ok("debug") => LevelFilter::Debug,
        Ok("trace") => LevelFilter::Trace,
        _ => LevelFilter::Info,
    };
    fern::Dispatch::new()
        .format(move |out, message, record| {
            let module_path = record.module_path().unwrap_or("<unknown>");
            // If stderr is a terminal, use colors when printing log level, otherwise use plain text
            let level = if is_terminal {
                colors.color(record.level()).to_string()
            } else {
                record.level().to_string()
            };
            out.finish(format_args!(
                "{} [{}] [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S%.3f"),
                level,
                module_path,
                message
            ))
        })
        .level(level_filter)
        .chain(std::io::stderr())
        .apply()?;
    Ok(())
}

#[pymodule]
fn _torchft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // setup logging on import
    setup_logging().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    m.add_class::<ManagerServer>()?;
    m.add_class::<ManagerClient>()?;
    m.add_class::<LighthouseServer>()?;
    m.add_class::<QuorumResult>()?;
    m.add_function(wrap_pyfunction!(lighthouse_main, m)?)?;

    Ok(())
}
