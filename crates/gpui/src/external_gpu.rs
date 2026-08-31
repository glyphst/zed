use std::{
    any::Any,
    fmt,
    sync::{
        Arc, Weak,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

static NEXT_EXTERNAL_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);
const EXTERNAL_TEXTURE_WRITE_RESERVED: u64 = 1 << 63;
const EXTERNAL_TEXTURE_SAMPLE_COUNT: u64 = !EXTERNAL_TEXTURE_WRITE_RESERVED;
const SAMPLE_PENDING: u8 = 0;
const SAMPLE_SUBMITTED: u8 = 1;
const SAMPLE_COMPLETE: u8 = 2;

#[derive(Default)]
struct ExternalTextureAccess {
    state: AtomicU64,
}

impl ExternalTextureAccess {
    fn try_reserve_sample(self: &Arc<Self>) -> Option<ExternalTextureSampleReservation> {
        let mut observed = self.state.load(Ordering::Acquire);
        loop {
            if observed & EXTERNAL_TEXTURE_WRITE_RESERVED != 0
                || observed & EXTERNAL_TEXTURE_SAMPLE_COUNT == EXTERNAL_TEXTURE_SAMPLE_COUNT
            {
                return None;
            }
            match self.state.compare_exchange_weak(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(ExternalTextureSampleReservation {
                        inner: Arc::new(ExternalTextureSampleReservationInner {
                            access: Arc::clone(self),
                            state: AtomicU8::new(SAMPLE_PENDING),
                        }),
                    });
                }
                Err(current) => observed = current,
            }
        }
    }

    fn try_reserve_write(self: &Arc<Self>) -> Option<ExternalTextureWriteReservation> {
        self.state
            .compare_exchange(
                0,
                EXTERNAL_TEXTURE_WRITE_RESERVED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| ExternalTextureWriteReservation {
                access: Arc::clone(self),
                active: true,
            })
    }

    fn release_sample(&self) {
        let previous = self.state.fetch_sub(1, Ordering::AcqRel);
        debug_assert_eq!(previous & EXTERNAL_TEXTURE_WRITE_RESERVED, 0);
        debug_assert_ne!(previous & EXTERNAL_TEXTURE_SAMPLE_COUNT, 0);
    }

    fn release_write(&self) {
        let previous = self
            .state
            .fetch_and(!EXTERNAL_TEXTURE_WRITE_RESERVED, Ordering::AcqRel);
        debug_assert_eq!(previous, EXTERNAL_TEXTURE_WRITE_RESERVED);
    }
}

struct ExternalTextureSampleReservationInner {
    access: Arc<ExternalTextureAccess>,
    state: AtomicU8,
}

impl ExternalTextureSampleReservationInner {
    fn complete(&self) {
        if self.state.swap(SAMPLE_COMPLETE, Ordering::AcqRel) != SAMPLE_COMPLETE {
            self.access.release_sample();
        }
    }
}

impl Drop for ExternalTextureSampleReservationInner {
    fn drop(&mut self) {
        self.complete();
    }
}

/// One pending or submitted scene use of an external texture.
///
/// This is an internal renderer hand-off exposed only so platform backends can
/// keep the texture read reservation alive through GPU completion.
#[doc(hidden)]
#[derive(Clone)]
pub struct ExternalTextureSampleReservation {
    inner: Arc<ExternalTextureSampleReservationInner>,
}

impl ExternalTextureSampleReservation {
    /// Claims the scene's pending reservation for one compositor submission.
    #[doc(hidden)]
    pub fn claim_submission(&self) -> Option<ExternalTextureSampleSubmission> {
        self.inner
            .state
            .compare_exchange(
                SAMPLE_PENDING,
                SAMPLE_SUBMITTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| ExternalTextureSampleSubmission {
                reservation: self.clone(),
            })
    }

    /// Acquires an independent reservation when the same scene is submitted again.
    #[doc(hidden)]
    pub fn fresh_submission(&self) -> Option<ExternalTextureSampleSubmission> {
        let reservation = self.inner.access.try_reserve_sample()?;
        reservation.claim_submission()
    }
}

impl fmt::Debug for ExternalTextureSampleReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalTextureSampleReservation")
            .field("state", &self.inner.state.load(Ordering::Acquire))
            .finish()
    }
}

/// Keeps one external-texture sample reserved until a compositor submission completes.
///
/// Platform backends move this value into their queue-completion callback. If
/// command recording or submission fails, dropping it releases the reservation.
#[doc(hidden)]
pub struct ExternalTextureSampleSubmission {
    reservation: ExternalTextureSampleReservation,
}

impl Drop for ExternalTextureSampleSubmission {
    fn drop(&mut self) {
        self.reservation.inner.complete();
    }
}

impl fmt::Debug for ExternalTextureSampleSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalTextureSampleSubmission")
            .finish_non_exhaustive()
    }
}

/// Exclusive permission to enqueue a write to an external texture.
///
/// Call [`Self::mark_submitted`] immediately after the write is submitted to
/// the same device queue. Later compositor samples are then ordered after that
/// write. Dropping an unsubmitted reservation safely cancels it.
pub struct ExternalTextureWriteReservation {
    access: Arc<ExternalTextureAccess>,
    active: bool,
}

impl ExternalTextureWriteReservation {
    /// Records that the reserved write has been submitted and releases the queue-order barrier.
    pub fn mark_submitted(mut self) {
        self.release();
    }

    fn release(&mut self) {
        if self.active {
            self.access.release_write();
            self.active = false;
        }
    }
}

impl Drop for ExternalTextureWriteReservation {
    fn drop(&mut self) {
        self.release();
    }
}

impl fmt::Debug for ExternalTextureWriteReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalTextureWriteReservation")
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

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
    access: Arc<ExternalTextureAccess>,
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
            access: Arc::new(ExternalTextureAccess::default()),
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
            access: Arc::downgrade(&self.access),
        }
    }

    /// Attempts to reserve this texture for an externally submitted write.
    ///
    /// The reservation fails while a GPUI scene references the texture or a
    /// previous compositor submission is still sampling it.
    pub fn try_reserve_write(&self) -> Option<ExternalTextureWriteReservation> {
        self.access.try_reserve_write()
    }

    /// Reserves one scene sample before it can race an external writer.
    #[doc(hidden)]
    pub fn try_reserve_sample(&self) -> Option<ExternalTextureSampleReservation> {
        self.access.try_reserve_sample()
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
    access: Weak<ExternalTextureAccess>,
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
        let payload = self.payload.upgrade()?;
        let access = self.access.upgrade()?;
        Some(ExternalTextureHandle {
            id: self.id,
            token: self.token,
            payload,
            access,
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

    #[test]
    fn scene_samples_and_external_writes_are_mutually_exclusive() {
        let texture = ExternalTextureHandle::new(ExternalGpuDeviceToken::new(10, 1), ());

        let sample = texture.try_reserve_sample().expect("scene sample");
        assert!(texture.try_reserve_write().is_none());
        let submission = sample.claim_submission().expect("first submission");
        assert!(sample.claim_submission().is_none());
        drop(submission);

        let write = texture.try_reserve_write().expect("write after sample");
        assert!(texture.try_reserve_sample().is_none());
        write.mark_submitted();
        assert!(texture.try_reserve_sample().is_some());
    }

    #[test]
    fn repeated_scene_submission_owns_an_independent_sample() {
        let texture = ExternalTextureHandle::new(ExternalGpuDeviceToken::new(11, 1), ());
        let sample = texture.try_reserve_sample().expect("scene sample");
        drop(sample.claim_submission().expect("first submission"));

        let repeated = sample.fresh_submission().expect("repeated submission");
        assert!(texture.try_reserve_write().is_none());
        drop(repeated);
        assert!(texture.try_reserve_write().is_some());
    }

    #[test]
    fn dropping_unsubmitted_reservations_releases_the_texture() {
        let texture = ExternalTextureHandle::new(ExternalGpuDeviceToken::new(12, 1), ());
        drop(texture.try_reserve_sample().expect("scene sample"));
        assert!(texture.try_reserve_write().is_some());

        let write = texture.try_reserve_write().expect("write reservation");
        drop(write);
        assert!(texture.try_reserve_sample().is_some());
    }
}
