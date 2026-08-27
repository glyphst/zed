use std::{
    any::Any,
    fmt,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_EXTERNAL_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);

/// Identifies one GPU device and its recovery generation.
///
/// External GPU resources are only valid while this token matches the token
/// exposed by the window that will present them. Device recovery increments
/// the generation so clients can discard resources created on the lost device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExternalGpuDeviceToken {
    identity: u64,
    generation: u64,
}

impl ExternalGpuDeviceToken {
    /// Creates a device token. Platform renderers should allocate identities
    /// that are unique for the lifetime of the process.
    pub const fn new(identity: u64, generation: u64) -> Self {
        Self {
            identity,
            generation,
        }
    }

    /// Returns the stable identity of the shared device context.
    pub const fn identity(self) -> u64 {
        self.identity
    }

    /// Returns the current device-recovery generation.
    pub const fn generation(self) -> u64 {
        self.generation
    }

    /// Returns this identity's next recovery generation.
    pub const fn next_generation(self) -> Self {
        Self {
            identity: self.identity,
            generation: self.generation.saturating_add(1),
        }
    }
}

/// An opaque platform GPU context that can be downcast by a matching backend.
///
/// GPUI itself does not expose a graphics API dependency through this type.
/// A backend crate, such as `gpui_wgpu`, supplies the concrete payload.
#[derive(Clone)]
pub struct ExternalGpuContext {
    token: ExternalGpuDeviceToken,
    payload: Arc<dyn Any + Send + Sync>,
}

impl ExternalGpuContext {
    /// Wraps a backend-owned context payload.
    pub fn new<T>(token: ExternalGpuDeviceToken, payload: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            token,
            payload: Arc::new(payload),
        }
    }

    /// Returns the device token associated with this context.
    pub const fn token(&self) -> ExternalGpuDeviceToken {
        self.token
    }

    /// Downcasts the opaque backend payload.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: Any + Send + Sync,
    {
        self.payload.downcast_ref()
    }
}

impl fmt::Debug for ExternalGpuContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalGpuContext")
            .field("token", &self.token)
            .finish_non_exhaustive()
    }
}

/// A process-local identifier for an externally owned texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExternalTextureId(u64);

impl ExternalTextureId {
    /// Returns the numeric process-local identifier.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// An opaque, reference-counted external texture handle.
///
/// The texture must have been created on the device identified by [`Self::token`].
/// The platform renderer validates the token and concrete payload before drawing.
#[derive(Clone)]
pub struct ExternalTextureHandle {
    id: ExternalTextureId,
    token: ExternalGpuDeviceToken,
    payload: Arc<dyn Any + Send + Sync>,
}

impl ExternalTextureHandle {
    /// Wraps a backend-owned texture payload and assigns it a stable local ID.
    pub fn new<T>(token: ExternalGpuDeviceToken, payload: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            id: ExternalTextureId(NEXT_EXTERNAL_TEXTURE_ID.fetch_add(1, Ordering::Relaxed)),
            token,
            payload: Arc::new(payload),
        }
    }

    /// Returns the process-local texture identifier.
    pub const fn id(&self) -> ExternalTextureId {
        self.id
    }

    /// Returns the GPU device token this texture was created with.
    pub const fn token(&self) -> ExternalGpuDeviceToken {
        self.token
    }

    /// Returns a non-owning handle that can be used to associate renderer-side
    /// caches with the lifetime of this external texture.
    pub fn downgrade(&self) -> WeakExternalTextureHandle {
        WeakExternalTextureHandle {
            id: self.id,
            token: self.token,
            payload: Arc::downgrade(&self.payload),
        }
    }

    /// Downcasts the opaque backend payload.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: Any + Send + Sync,
    {
        self.payload.downcast_ref()
    }
}

/// A non-owning reference to an external texture handle.
///
/// Platform renderers use this to keep derived resources, such as bind groups,
/// alive while the application retains the texture without making those
/// derived resources extend the texture's lifetime.
#[derive(Clone)]
pub struct WeakExternalTextureHandle {
    id: ExternalTextureId,
    token: ExternalGpuDeviceToken,
    payload: Weak<dyn Any + Send + Sync>,
}

impl WeakExternalTextureHandle {
    /// Returns the process-local texture identifier.
    pub const fn id(&self) -> ExternalTextureId {
        self.id
    }

    /// Returns the GPU device token this texture was created with.
    pub const fn token(&self) -> ExternalGpuDeviceToken {
        self.token
    }

    /// Upgrades this reference while the external texture is still retained.
    pub fn upgrade(&self) -> Option<ExternalTextureHandle> {
        self.payload.upgrade().map(|payload| ExternalTextureHandle {
            id: self.id,
            token: self.token,
            payload,
        })
    }

    /// Returns whether an owning handle still exists.
    pub fn is_alive(&self) -> bool {
        self.payload.strong_count() != 0
    }
}

impl fmt::Debug for WeakExternalTextureHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeakExternalTextureHandle")
            .field("id", &self.id)
            .field("token", &self.token)
            .field("alive", &self.is_alive())
            .finish()
    }
}

impl fmt::Debug for ExternalTextureHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalTextureHandle")
            .field("id", &self.id)
            .field("token", &self.token)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_payloads_are_typed_and_generation_scoped() {
        let token = ExternalGpuDeviceToken::new(7, 2);
        let context = ExternalGpuContext::new(token, String::from("context"));
        let texture = ExternalTextureHandle::new(token, 42_u32);

        assert_eq!(context.token(), token);
        assert_eq!(
            context.downcast_ref::<String>().map(String::as_str),
            Some("context")
        );
        assert!(context.downcast_ref::<u32>().is_none());
        assert_eq!(texture.token(), token);
        assert_eq!(texture.downcast_ref::<u32>(), Some(&42));
        assert_eq!(token.next_generation(), ExternalGpuDeviceToken::new(7, 3));
    }

    #[test]
    fn weak_texture_handles_do_not_extend_texture_lifetime() {
        let token = ExternalGpuDeviceToken::new(9, 1);
        let texture = ExternalTextureHandle::new(token, 42_u32);
        let weak = texture.downgrade();

        assert_eq!(weak.id(), texture.id());
        assert_eq!(weak.token(), token);
        assert!(weak.is_alive());
        assert_eq!(
            weak.upgrade()
                .and_then(|texture| texture.downcast_ref::<u32>().copied()),
            Some(42)
        );

        drop(texture);

        assert!(!weak.is_alive());
        assert!(weak.upgrade().is_none());
    }
}
