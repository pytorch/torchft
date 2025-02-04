// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use _torchft::lighthouse::{Lighthouse, LighthouseOpt};
use structopt::StructOpt;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    let opt = LighthouseOpt::from_args();
    let lighthouse = Lighthouse::new(opt).await.unwrap();

    lighthouse.run().await.unwrap();
}
