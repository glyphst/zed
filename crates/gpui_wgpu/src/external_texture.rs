use gpui::{ExternalGpuContext, ExternalGpuDeviceToken, ExternalTextureHandle};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// The concrete WGPU payload carried by [`ExternalGpuContext`].
///
/// Call [`Self::from_gpui`] to downcast the runtime-neutral GPUI handle, then
/// use the cloned device and queue to create resources that GPUI can sample
/// without a CPU readback or cross-device copy.
#[derive(Clone)]
pub struct WgpuExternalContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    token: ExternalGpuDeviceToken,
    device_lost: Arc<AtomicBool>,
}

impl WgpuExternalContext {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        token: ExternalGpuDeviceToken,
        device_lost: Arc<AtomicBool>,
    ) -> Self {
        Self {
            device,
            queue,
            token,
            device_lost,
        }
    }

    /// Downcasts a runtime-neutral GPUI context to its WGPU payload.
    pub fn from_gpui(context: &ExternalGpuContext) -> Option<&Self> {
        context.downcast_ref()
    }

    /// Returns the shared WGPU device used by the GPUI window.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Returns the shared WGPU queue used by the GPUI window.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Returns the device identity and recovery generation.
    pub const fn token(&self) -> ExternalGpuDeviceToken {
        self.token
    }

    /// Returns whether GPUI has observed this WGPU device being lost.
    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Relaxed)
    }

    /// Wraps a texture view created on this context's device for ordered GPUI
    /// presentation. The view must be a filterable, non-multisampled 2D float
    /// texture view.
    pub fn texture_handle(&self, view: Arc<wgpu::TextureView>) -> ExternalTextureHandle {
        ExternalTextureHandle::new(self.token, WgpuExternalTexture { view })
    }
}

pub(crate) struct WgpuExternalTexture {
    pub(crate) view: Arc<wgpu::TextureView>,
}
