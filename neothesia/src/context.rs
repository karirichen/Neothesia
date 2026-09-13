use std::sync::Arc;

use crate::{
    NeothesiaEvent, TransformUniform, config::Config, input_manager::InputManager,
    output_manager::OutputManager, utils::window::WindowState,
};
use neothesia_core::render::{QuadRendererFactory, TextRendererFactory};
use wgpu_jumpstart::{Gpu, Uniform};
use winit::event_loop::EventLoopProxy;

use winit::window::Window;

pub struct Context {
    pub window: Arc<Window>,

    pub window_state: WindowState,
    pub gpu: Gpu,

    pub transform: Uniform<TransformUniform>,
    pub text_renderer_factory: TextRendererFactory,
    pub quad_renderer_factory: QuadRendererFactory,

    pub output_manager: OutputManager,
    /// Live microphone input connection, when enabled (Phase 4 wires
    /// the settings UI; startup restore also Phase 4).
    pub audio_input: Option<audio_input::AudioInputConnection>,
    /// Pitches the game itself is sounding; used to suppress
    /// mic-detected ghost notes (echo suppression, design §6).
    pub sounding: std::sync::Arc<crate::sounding_tracker::SharedSoundingTracker>,
    pub input_manager: InputManager,
    pub config: Config,

    pub proxy: EventLoopProxy<NeothesiaEvent>,

    /// Last frame timestamp
    pub frame_timestamp: std::time::Instant,

    #[cfg(debug_assertions)]
    pub fps_ticker: neothesia_core::utils::fps_ticker::Fps,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.config.save();
    }
}

impl Context {
    pub fn new(
        window: Arc<Window>,
        window_state: WindowState,
        proxy: EventLoopProxy<NeothesiaEvent>,
        gpu: Gpu,
    ) -> Self {
        let transform_uniform = Uniform::new(
            &gpu.device,
            TransformUniform::default(),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );

        let config = Config::new();

        let text_renderer_factory = TextRendererFactory::new(&gpu);
        let quad_renderer_factory = QuadRendererFactory::new(&gpu, &transform_uniform);

        Self {
            window,

            window_state,
            gpu,
            transform: transform_uniform,
            text_renderer_factory,
            quad_renderer_factory,

            output_manager: Default::default(),
            audio_input: None,
            sounding: crate::sounding_tracker::shared(),
            input_manager: InputManager::new(proxy.clone()),
            config,
            proxy,
            frame_timestamp: std::time::Instant::now(),

            #[cfg(debug_assertions)]
            fps_ticker: Default::default(),
        }
    }

    pub fn resize(&mut self) {
        self.transform.data.update(
            self.window_state.physical_size.width as f32,
            self.window_state.physical_size.height as f32,
            self.window_state.scale_factor as f32,
        );
        self.transform.update(&self.gpu.queue);
    }

    /// Establish/re-establish the microphone input connection.
    /// The caller must ensure the model is in place (Phase 4 handles
    /// download; this task always calls it with an existing path).
    pub fn connect_audio_input(&mut self, model_path: &std::path::Path) -> Result<(), String> {
        self.audio_input = None; // drop the old connection

        let device_name = match self.config.mic_device() {
            Some(name) => Some(name.to_owned()),
            // Phase 5 validates device choice on real hardware;
            // first-enumerated may be an aggregate on macOS.
            None => audio_input::AudioInputManager::devices()
                .first()
                .cloned()
                .map(|d| d.0),
        };

        let Some(device_name) = device_name else {
            return Err("no microphone devices found".into());
        };

        let proxy = self.proxy.clone();
        let device = audio_input::MicDevice(device_name);
        let model_path = model_path.to_owned();

        let conn = audio_input::AudioInputManager::connect(&device, &model_path, move |event| {
            if let Some(ev) = crate::mic_event_to_neothesia(event) {
                // Send errors after loop teardown are expected and
                // harmless (late events for ~60ms after drop).
                proxy.send_event(ev).ok();
            }
        })
        .map_err(|e| e.to_string())?;

        self.audio_input = Some(conn);
        Ok(())
    }

    pub fn disconnect_audio_input(&mut self) {
        self.audio_input = None;
    }
}
