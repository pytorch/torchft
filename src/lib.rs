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

use core::time::Duration;
use std::cmp;
use std::env;
use std::sync::Arc;
use std::thread::available_parallelism;

use anyhow::Result;
use atty::Stream;
use chrono::Local;
use fern::colors::Color;
use fern::colors::ColoredLevelConfig;
use log::LevelFilter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTimeoutError;
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tonic::Status;
use tonic::transport::Channel;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyString;

use crate::torchftpb::CheckpointMetadataRequest;
use crate::torchftpb::LighthouseHeartbeatRequest;
use crate::torchftpb::LighthouseQuorumRequest;
use crate::torchftpb::ManagerQuorumRequest;
use crate::torchftpb::ShouldCommitRequest;
use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;
use crate::torchftpb::manager_service_client::ManagerServiceClient;

// Get the number of threads to use for the tokio runtime
fn num_threads() -> usize {
    let default_threads = 4;
    let num_cpus = available_parallelism()
        .and_then(|p| Ok(p.get()))
        .unwrap_or(default_threads);

    let num_threads = env::var("TOKIO_WORKER_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cmp::min(default_threads, num_cpus));

    num_threads
}

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
///     quorum_retries (int): The number of retries for quorum requests to lighthouse server.
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
        quorum_retries: i64,
    ) -> PyResult<Self> {
        py.allow_threads(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_threads())
                .thread_name("torchft-manager")
                .enable_all()
                .build()?;
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
                    quorum_retries,
                ))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let handle = runtime.spawn(manager.clone().run());
            Ok(Self {
                handle,
                manager,
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
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_threads())
                .thread_name("torchft-mgrclnt")
                .enable_all()
                .build()?;
            let client = runtime
                .block_on(manager::manager_client_new(addr, connect_timeout))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(Self { runtime, client })
        })
    }

    fn _quorum(
        &self,
        py: Python<'_>,
        group_rank: i64,
        step: i64,
        checkpoint_metadata: String,
        shrink_only: bool,
        init_sync: bool,
        commit_failures: i64,
        timeout: Duration,
    ) -> Result<QuorumResult, StatusError> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(ManagerQuorumRequest {
                group_rank,
                step,
                checkpoint_metadata,
                shrink_only,
                init_sync,
                commit_failures,
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
                recover_src_replica_rank: resp.recover_src_replica_rank,
                recover_dst_replica_ranks: resp.recover_dst_replica_ranks,
                store_address: resp.store_address,
                max_step: resp.max_step,
                max_replica_rank: resp.max_replica_rank,
                max_world_size: resp.max_world_size,
                heal: resp.heal,
                replica_ids: resp.replica_ids,
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
            let mut request = tonic::Request::new(CheckpointMetadataRequest { rank });

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
        group_rank: i64,
        step: i64,
        should_commit: bool,
        timeout: Duration,
    ) -> Result<bool, StatusError> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(ShouldCommitRequest {
                group_rank,
                step,
                should_commit,
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
    recover_src_replica_rank: Option<i64>,
    recover_dst_replica_ranks: Vec<i64>,
    store_address: String,
    max_step: i64,
    max_replica_rank: Option<i64>,
    max_world_size: i64,
    heal: bool,
    replica_ids: Vec<String>,
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
            recover_src_replica_rank: None,
            recover_dst_replica_ranks: Vec::new(),
            store_address: "".to_string(),
            max_step: 0,
            max_replica_rank: None,
            max_world_size: 1,
            heal: false,
            replica_ids: Vec::new(),
        }
    }
}

fn reset_python_signals(py: Python<'_>) -> PyResult<()> {
    // clear python signal handlers
    // signal.signal(signal.SIGINT, signal.SIG_DFL)
    let signal = py.import("signal")?;
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
    let rt = tokio::runtime::Builder::new_multi_thread()
        .thread_name("torchft-lighths")
        .worker_threads(num_threads())
        .enable_all()
        .build()?;
    rt.block_on(lighthouse_main_async(opt))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

async fn lighthouse_main_async(opt: lighthouse::LighthouseOpt) -> Result<()> {
    let lighthouse = lighthouse::Lighthouse::new(opt).await?;

    lighthouse.run().await?;

    Ok(())
}

/// quorum member of one quorum.
///
/// Args:
///     replica_id (str): The string id of the replica calling quorum.
///     address (str): The address of the replica calling quorum.
///     store_address (str): The address of the store.
///     step (int): The step of the replica calling quorum.
///     world_size (int): The world size of the replica calling quorum.
///     shrink_only (bool): Whether the quorum is for shrinking only.
///     timeout (timedelta): The timeout for quorum.
///     data (dict or None): The data to be passed with quorum.
#[pyclass(get_all, set_all)]
pub struct QuorumMember {
    replica_id: String,
    address: String,
    store_address: String,
    step: i64,
    world_size: u64,
    shrink_only: bool,
    data: Option<Py<PyDict>>,
}

impl QuorumMember {
    // PyDict has not implemeted Clone, so we need to implement it manually
    pub fn clone_with_py(&self, py: Python) -> Self {
        QuorumMember {
            replica_id: self.replica_id.clone(),
            address: self.address.clone(),
            store_address: self.store_address.clone(),
            step: self.step,
            world_size: self.world_size,
            shrink_only: self.shrink_only,
            data: self.data.as_ref().map(|d| d.clone_ref(py)),
        }
    }
}

impl Clone for QuorumMember {
    fn clone(&self) -> Self {
        Python::with_gil(|py| self.clone_with_py(py))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct Timestamp {
    pub seconds: i64,
    pub nanos: i32,
}

/// quorum result.
///
/// Args:
///     quorum_id (int): The id of current quorum.
///     participants (list[QuorumMember]): All members within the quorum.
///     created (timedelta): Time of quorum created in server.
#[pyclass(get_all, set_all)]
struct Quorum {
    quorum_id: i64,
    participants: Vec<QuorumMember>,
    created: Timestamp,
}

impl From<prost_types::Timestamp> for Timestamp {
    fn from(ts: prost_types::Timestamp) -> Self {
        Timestamp {
            seconds: ts.seconds,
            nanos: ts.nanos,
        }
    }
}

// Util functions to convert between python dict and rust string using json.
fn pydict_to_string<'py>(py: Python, data: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
    match data {
        Some(d) => {
            let json = py.import("json")?;
            let json_obj = json.call_method1("dumps", (d,))?;
            let py_str: &Bound<PyString> = json_obj.downcast()?;
            Ok(py_str.to_str()?.to_owned())
        }
        None => Ok(String::new()),
    }
}

fn string_to_pydict(py: Python, s: &str) -> PyResult<Option<Py<PyDict>>> {
    if s.is_empty() {
        return Ok(None); // Treat empty string as None
    }

    let json = py.import("json")?;
    let obj = json.call_method1("loads", (s,))?;
    let dict: &Bound<PyDict> = obj.downcast()?;
    Ok(Some(dict.to_owned().into())) // convert Bound<PyDict> -> Py<PyDict>
}

fn convert_quorum_member(py: Python, m: &torchftpb::QuorumMember) -> PyResult<QuorumMember> {
    Ok(QuorumMember {
        replica_id: m.replica_id.clone(),
        address: m.address.clone(),
        store_address: m.store_address.clone(),
        step: m.step.clone(),
        world_size: m.world_size.clone(),
        shrink_only: m.shrink_only.clone(),
        data: string_to_pydict(py, &m.data)?,
    })
}

fn convert_quorum(py: Python, q: &torchftpb::Quorum) -> PyResult<Quorum> {
    let participants: Vec<QuorumMember> = q
        .participants
        .iter()
        .map(|m| convert_quorum_member(py, m)) // this expects &m
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Quorum {
        quorum_id: q.quorum_id,
        participants,
        created: Timestamp::from(q.created.unwrap()),
    })
}

/// LighthouseClient is a GRPC client to the lighthouse service.
///
/// It is used to directly communicate with the lighthouse Server.
///
/// Args:
///     addr (str): The HTTP address of the lighthouse server.
///     connect_timeout (timedelta): The timeout for connecting to the lighthouse server.
#[pyclass]
struct LighthouseClient {
    client: LighthouseServiceClient<Channel>,
    runtime: Runtime,
}

#[pymethods]
impl LighthouseClient {
    #[pyo3(signature = (addr, connect_timeout))]
    #[new]
    fn new(py: Python<'_>, addr: String, connect_timeout: Duration) -> PyResult<Self> {
        py.allow_threads(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_threads())
                .thread_name("torchft-lhclnt")
                .enable_all()
                .build()?;
            let client = runtime
                .block_on(manager::lighthouse_client_new(addr, connect_timeout))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { client, runtime })
        })
    }

    /// quorum sends a request to the lighthouse server to form a quorum.
    ///
    /// Args:
    ///     replica_id (str): The string id of the replica calling quorum.
    ///     timeout (timedelta): The timeout for quorum.
    ///     address (str): The address of the replica calling quorum. Default: "".
    ///     store_address (str): The address of the store. Default: "".
    ///     step (int): The step of the replica calling quorum. Default: 0.
    ///     world_size (int): The world size of the replica calling quorum. Default: 0.
    ///     shrink_only (bool): Whether the quorum is for shrinking only. Default: false.
    ///     data (Optional[dict]): The data to be passed with quorum.
    ///
    /// Returns:
    ///     Quorum: Current quorum if successful.
    #[pyo3(signature = (
        replica_id,
        timeout,
        address = "".to_string(),
        store_address = "".to_string(),
        step = 0,
        world_size = 0,
        shrink_only = false,
        data = None
    ))]
    fn quorum<'py>(
        &self,
        py: Python<'_>,
        replica_id: String,
        timeout: Duration,
        address: String,
        store_address: String,
        step: i64,
        world_size: u64,
        shrink_only: bool,
        data: Option<&Bound<'_, PyDict>>,
    ) -> Result<Quorum, StatusError> {
        let data_string = pydict_to_string(py, data)?;
        let quorum: Result<torchftpb::Quorum, StatusError> = py.allow_threads(move || {
            let mut request = tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(torchftpb::QuorumMember {
                    replica_id,
                    address,
                    store_address,
                    step,
                    world_size,
                    shrink_only,
                    data: data_string,
                    commit_failures: 0,
                }),
            });

            // This timeout is processed on the server side so we also enable
            // keep alives to detect server health.
            request.set_timeout(timeout);

            let response = self.runtime.block_on(self.client.clone().quorum(request))?;
            let resp = response.into_inner();
            let quorum = resp
                .quorum
                .ok_or_else(|| Status::internal("missing quorum"))?;
            Ok(quorum)
        });
        Ok(convert_quorum(py, &quorum?)?)
    }

    /// Send a single heartbeat to the lighthouse.
    ///
    /// Args:
    ///     replica_id (str):  The replica_id you registered with.
    ///     timeout      (timedelta, optional):  Per-RPC deadline.  Default = 5 s.
    #[pyo3(signature = (replica_id, timeout = Duration::from_secs(5)))]
    fn heartbeat(
        &self,
        py: Python<'_>,
        replica_id: String,
        timeout: Duration,
    ) -> Result<(), StatusError> {
        py.allow_threads(move || {
            let mut req = tonic::Request::new(LighthouseHeartbeatRequest { replica_id });
            req.set_timeout(timeout);
            self.runtime.block_on(self.client.clone().heartbeat(req))?;
            Ok(())
        })
    }
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
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(num_threads())
                .thread_name("torchft-lighths")
                .enable_all()
                .build()?;

            let lighthouse = rt
                .block_on(lighthouse::Lighthouse::new(lighthouse::LighthouseOpt {
                    bind,
                    min_replicas,
                    join_timeout_ms,
                    quorum_tick_ms,
                    heartbeat_timeout_ms,
                }))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(Self {
                handle: rt.spawn(lighthouse.clone().run()),
                lighthouse,
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

impl From<pyo3::PyErr> for StatusError {
    fn from(err: pyo3::PyErr) -> Self {
        StatusError(Status::internal(err.to_string()))
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
        Ok(value) => {
            let value_lower = value.to_lowercase();
            match value_lower.as_str() {
                "error" => LevelFilter::Error,
                "warn" => LevelFilter::Warn,
                "info" => LevelFilter::Info,
                "debug" => LevelFilter::Debug,
                "trace" => LevelFilter::Trace,
                _ => LevelFilter::Info,
            }
        }
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

    m.add_class::<Timestamp>()?;
    m.add_class::<QuorumMember>()?;
    m.add_class::<Quorum>()?;
    m.add_class::<ManagerServer>()?;
    m.add_class::<ManagerClient>()?;
    m.add_class::<LighthouseServer>()?;
    m.add_class::<LighthouseClient>()?;
    m.add_class::<QuorumResult>()?;
    m.add_function(wrap_pyfunction!(lighthouse_main, m)?)?;

    Ok(())
}
