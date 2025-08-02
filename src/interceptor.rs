use tonic::{metadata::MetadataValue, service::Interceptor, Request, Status};

/// Attaches user-assigned room-id header to every outbound RPC
#[derive(Clone)]
pub struct RoomIdInterceptor {
    room: String,
}

impl RoomIdInterceptor {
    pub fn new(room: String) -> Self {
        Self { room }
    }
}

impl Interceptor for RoomIdInterceptor {
    fn call(&mut self, mut req: Request<()>) -> Result<Request<()>, Status> {
        req.metadata_mut().insert(
            crate::router::ROOM_ID_HEADER,
            MetadataValue::try_from(self.room.as_str()).expect("ascii header"),
        );
        Ok(req)
    }
}
