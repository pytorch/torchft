// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::net::SocketAddr;
use structopt::StructOpt;
use tonic::transport::Server;
use torchft::lighthouse::LighthouseOpt;
use torchft::torchftpb::lighthouse_service_server::LighthouseServiceServer;
use torchft::router::Router;


#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let opt = LighthouseOpt::from_args();
    let router = Router::new(opt.clone());
    Server::builder()
        .add_service(LighthouseServiceServer::new(router))
        .serve(opt.bind.parse::<SocketAddr>().unwrap())
        .await
        .unwrap();
}
