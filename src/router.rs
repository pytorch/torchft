use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use tonic::{Request, Response, Status};

use crate::{
    lighthouse::{Lighthouse, LighthouseOpt},
    torchftpb::{
        lighthouse_service_server::LighthouseService, LighthouseHeartbeatRequest,
        LighthouseHeartbeatResponse, LighthouseQuorumRequest, LighthouseQuorumResponse,
    },
};

/// Metadata header for both client and router
pub const ROOM_ID_HEADER: &str = "room-id";

/// Top-level service registered with tonic’s `Server::builder()`
#[derive(Clone)]
pub struct Router {
    rooms: Arc<DashMap<String, Arc<Lighthouse>>>,
    tmpl_opt: LighthouseOpt, // (cloned for each new room)
}

/// Designates a single tonic gRPC server into many logical “rooms.”
/// Inspects the `room-id` metadata header on each request, then
/// lazily creates or reuses an Arc<Lighthouse> for that namespace
impl Router {
    /// Create a new router given the CLI/config options that are
    /// normally passed straight to `Lighthouse::new`.
    pub fn new(tmpl_opt: LighthouseOpt) -> Self {
        Self {
            rooms: Arc::new(DashMap::new()),
            tmpl_opt,
        }
    }

    /// Room lookup: creation if it doesn't exist, access if it does
    async fn room(&self, id: &str) -> Arc<Lighthouse> {
        // 1. Quick optimistic read (no locking contention).
        if let Some(handle) = self.rooms.get(id) {
            return handle.clone();
        }

        // 2. Build the Lighthouse instance *off the map* so
        //    we don't hold any guard across `.await`.
        let new_room = Lighthouse::new(self.tmpl_opt.clone())
            .await
            .expect("failed to create Lighthouse");

        // 3. Second pass: insert if still vacant, otherwise reuse
        //    whatever another task inserted first.
        match self.rooms.entry(id.to_owned()) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                entry.insert(new_room.clone());
                new_room
            }
        }
    }

    /// Extracts `"room-id"` from metadata, defaulting to `"default"`.
    fn extract_room_id(meta: &tonic::metadata::MetadataMap) -> &str {
        meta.get(ROOM_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("default")
    }
}

#[tonic::async_trait]
impl LighthouseService for Router {
    async fn quorum(
        &self,
        req: Request<LighthouseQuorumRequest>,
    ) -> Result<Response<LighthouseQuorumResponse>, Status> {
        let id = Self::extract_room_id(req.metadata()).to_owned();
        let room = self.room(&id).await;
        <Arc<Lighthouse> as LighthouseService>::quorum(&room, req).await
    }

    async fn heartbeat(
        &self,
        req: Request<LighthouseHeartbeatRequest>,
    ) -> Result<Response<LighthouseHeartbeatResponse>, Status> {
        let id = Self::extract_room_id(req.metadata()).to_owned();
        let room = self.room(&id).await;
        <Arc<Lighthouse> as LighthouseService>::heartbeat(&room, req).await
    }
}
