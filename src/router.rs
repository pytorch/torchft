use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use dashmap::{mapref::entry::Entry, DashMap};
use futures::FutureExt;
use tonic::{
    body::BoxBody,
    codegen::http::{HeaderMap, Request, Response},
    server::NamedService,
};
use tower::Service;

use crate::{
    lighthouse::{Lighthouse, LighthouseOpt},
    torchftpb::lighthouse_service_server::LighthouseServiceServer,
};

/// Metadata header recognised by both client interceptor and this router.
pub const ROOM_ID_HEADER: &str = "room-id";

/// gRPC server for a single room (inner state = `Arc<Lighthouse>`).
type GrpcSvc = LighthouseServiceServer<Arc<Lighthouse>>;

#[derive(Clone)]
pub struct Router {
    rooms: Arc<DashMap<String, Arc<Lighthouse>>>,
    tmpl_opt: LighthouseOpt,
}

impl Router {
    pub fn new(tmpl_opt: LighthouseOpt) -> Self {
        Self {
            rooms: Arc::new(DashMap::new()),
            tmpl_opt,
        }
    }

    fn room_id(hdrs: &HeaderMap) -> &str {
        hdrs.get(ROOM_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("default")
    }

    async fn room_service(
        rooms: Arc<DashMap<String, Arc<Lighthouse>>>,
        tmpl: LighthouseOpt,
        id: &str,
    ) -> Arc<Lighthouse> {
        if let Some(lh) = rooms.get(id) {
            return lh.clone();
        }

        let lh = Lighthouse::new(id.to_owned(), tmpl.clone())
            .await
            .expect("failed to create Lighthouse");

        match rooms.entry(id.to_owned()) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(v) => {
                v.insert(lh.clone());
                lh
            }
        }
    }
}

// Tower::Service implementation
impl Service<Request<BoxBody>> for Router {
    type Response = Response<BoxBody>;
    type Error = Infallible;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<BoxBody>) -> Self::Future {
        let rooms = self.rooms.clone();
        let tmpl = self.tmpl_opt.clone();
        let room = Self::room_id(req.headers()).to_owned();

        async move {
            let lh = Self::room_service(rooms, tmpl, &room).await;

            let mut svc = LighthouseServiceServer::new(lh);
            let resp = svc
                .call(req)
                .await
                .map_err(|_e| -> Infallible { unreachable!() })?;

            Ok(resp)
        }
        .boxed()
    }
}

// Forward tonic’s NamedService marker
impl NamedService for Router {
    const NAME: &'static str = <GrpcSvc as NamedService>::NAME;
}
