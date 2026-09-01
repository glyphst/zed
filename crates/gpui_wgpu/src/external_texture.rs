use gpui::{
    Bounds, ContentMask, ExternalGpuContext, ExternalGpuDeviceToken, ExternalGpuRenderHandle,
    ExternalTextureHandle, ScaledPixels,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(not(target_family = "wasm"))]
use std::{
    sync::mpsc::{SyncSender, sync_channel},
    thread,
    time::Duration,
};

/// One device-wide worker that drives WGPU queue-completion callbacks away
/// from GPUI's paint path.
#[derive(Clone)]
pub(crate) struct WgpuCompletionPoller {
    #[cfg(not(target_family = "wasm"))]
    wake: SyncSender<()>,
}

impl WgpuCompletionPoller {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        device_lost: Arc<AtomicBool>,
    ) -> anyhow::Result<Self> {
        let (wake, requests) = sync_channel(1);
        thread::Builder::new()
            .name("gpui-wgpu-completion".into())
            .spawn(move || {
                while requests.recv().is_ok() {
                    // Let the producer finish registering callbacks and give
                    // another same-device submitter a short opportunity to
                    // enqueue its ordered work before the blocking poll.
                    thread::sleep(Duration::from_micros(500));
                    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                        device_lost.store(true, Ordering::Release);
                    }
                }
            })
            .map_err(|error| anyhow::anyhow!(error))?;
        Ok(Self { wake })
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn new(
        _device: Arc<wgpu::Device>,
        _device_lost: Arc<AtomicBool>,
    ) -> anyhow::Result<Self> {
        // Browser event loops drive WebGPU completion callbacks.
        Ok(Self {})
    }

    pub(crate) fn request(&self) {
        #[cfg(not(target_family = "wasm"))]
        {
            // A single queued wake is enough: a blocking poll drains all work
            // submitted before it begins, and a racing submit can enqueue the
            // next wake after the receiver consumes this one.
            let _ = self.wake.try_send(());
        }
    }
}

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
    surface_format: wgpu::TextureFormat,
    device_lost: Arc<AtomicBool>,
    completion_poller: WgpuCompletionPoller,
}

impl WgpuExternalContext {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        token: ExternalGpuDeviceToken,
        surface_format: wgpu::TextureFormat,
        device_lost: Arc<AtomicBool>,
        completion_poller: WgpuCompletionPoller,
    ) -> Self {
        Self {
            device,
            queue,
            token,
            surface_format,
            device_lost,
            completion_poller,
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

    /// Returns the color format of the current GPUI presentation target.
    pub const fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Returns whether GPUI has observed this WGPU device being lost.
    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Relaxed)
    }

    /// Wakes GPUI's device-wide completion worker after an external queue submission.
    ///
    /// Register any `Queue::on_submitted_work_done` callback before calling
    /// this method. The worker may block on the backend, but the UI paint path
    /// never does.
    pub fn request_completion_poll(&self) {
        self.completion_poller.request();
    }

    /// Wraps a texture view created on this context's device for ordered GPUI
    /// presentation. The view must be a filterable, non-multisampled 2D float
    /// texture view.
    pub fn texture_handle(&self, view: Arc<wgpu::TextureView>) -> ExternalTextureHandle {
        ExternalTextureHandle::new(self.token, WgpuExternalTexture { view })
    }

    /// Wraps a WGPU callback for painter-ordered execution in GPUI's frame encoder.
    pub fn render_handle(&self, render: Arc<dyn WgpuExternalRender>) -> ExternalGpuRenderHandle {
        ExternalGpuRenderHandle::new(self.token, WgpuExternalRenderPayload { render })
    }
}

/// The current GPUI WGPU target exposed to one painter-ordered external renderer.
pub struct WgpuExternalRenderContext<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target: &'a wgpu::TextureView,
    pub target_format: wgpu::TextureFormat,
    pub target_size: [u32; 2],
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

/// Records trusted same-device commands into GPUI's current frame.
pub trait WgpuExternalRender: Send + Sync {
    fn render(&self, context: &mut WgpuExternalRenderContext<'_>) -> anyhow::Result<()>;
}

pub(crate) struct WgpuExternalRenderPayload {
    pub(crate) render: Arc<dyn WgpuExternalRender>,
}

pub(crate) struct WgpuExternalTexture {
    pub(crate) view: Arc<wgpu::TextureView>,
}
