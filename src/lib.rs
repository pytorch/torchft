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
use std::env;
use std::sync::Arc;

use anyhow::Result;
use log::info;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tonic::Status;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

use crate::net::Channel;
use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::{CheckpointMetadataRequest, ManagerQuorumRequest, ShouldCommitRequest};
use pyo3::prelude::*;

#[pyclass]
struct Manager {
    handle: JoinHandle<Result<()>>,
    manager: Arc<manager::Manager>,
    _runtime: Runtime,
}

#[pymethods]
impl Manager {
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

    fn address(&self) -> PyResult<String> {
        Ok(self.manager.address().to_string())
    }

    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(move || {
            self.handle.abort();
        })
    }
}

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

    fn quorum(
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

    fn checkpoint_metadata(
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

#[pyclass]
struct Lighthouse {
    lighthouse: Arc<lighthouse::Lighthouse>,
    handle: JoinHandle<Result<()>>,
    _runtime: Runtime,
}

#[pymethods]
impl Lighthouse {
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

    fn address(&self) -> PyResult<String> {
        Ok(self.lighthouse.address().to_string())
    }

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

fn init_logging() -> PyResult<()> {
    // setup logging on import
    let mut log = stderrlog::new();
    log.verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond);

    if env::var("CLICOLOR_FORCE").is_ok() {
        log.color(stderrlog::ColorChoice::AlwaysAnsi);
    }

    log.init()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(())
}

fn init_tracing() -> PyResult<()> {
    use opentelemetry::trace::Tracer;
    use opentelemetry::trace::TracerProvider as OpenTelemetryTracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::TracerProvider;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::{filter::EnvFilter, Layer};

    fn set_tracer_provider(tracer_provider: TracerProvider) -> PyResult<()> {
        opentelemetry::global::set_tracer_provider(tracer_provider.clone());

        let layer = tracing_opentelemetry::layer()
            .with_error_records_to_exceptions(true)
            .with_tracer(tracer_provider.tracer(""));

        // Create a new tracing::Fmt layer to print the logs to stdout. It has a
        // default filter of `info` level and above, and `debug` and above for logs
        // from OpenTelemetry crates. The filter levels can be customized as needed.
        let filter_fmt =
            EnvFilter::new("info").add_directive("opentelemetry=debug".parse().unwrap());
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_thread_names(true)
            .with_filter(filter_fmt);

        let subscriber = tracing_subscriber::registry().with(fmt_layer).with(layer);
        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        info!("OpenTelemetry tracing enabled");

        Ok(())
    }

    match env::var("TORCHFT_OTEL_OTLP") {
        Ok(endpoint) => {
            let runtime = Runtime::new()?;

            runtime.block_on(async move {
                info!("Enabling OpenTelemetry OTLP with {}", endpoint);
                let exporter = opentelemetry_otlp::SpanExporter::builder()
                    .with_tonic()
                    .with_endpoint(endpoint)
                    .with_timeout(Duration::from_secs(10))
                    .build()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                let tracer_provider = TracerProvider::builder()
                    .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
                    .build();

                set_tracer_provider(tracer_provider)?;

                Ok::<(), pyo3::PyErr>(())
            })?;
        }
        Err(_) => {}
    };
    match env::var("TORCHFT_OTEL_STDOUT") {
        Ok(_) => {
            info!("Enabling OpenTelemetry stdout");
            let exporter = opentelemetry_stdout::SpanExporter::default();
            let tracer_provider = TracerProvider::builder()
                .with_simple_exporter(exporter)
                .build();

            set_tracer_provider(tracer_provider)?;
        }
        Err(_) => {}
    }

    let tracer = opentelemetry::global::tracer("my_tracer");
    tracer.in_span("doing_work", |cx| {
        // Traced app logic here...
    });

    Ok(())
}

#[pymodule]
fn torchft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_logging()?;
    init_tracing()?;

    m.add_class::<Manager>()?;
    m.add_class::<ManagerClient>()?;
    m.add_class::<Lighthouse>()?;
    m.add_class::<QuorumResult>()?;
    m.add_function(wrap_pyfunction!(lighthouse_main, m)?)?;

    Ok(())
}
