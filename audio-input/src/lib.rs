//! Microphone capture + streaming polyphonic pitch detection.
//!
//! Public API mirrors `midi_io`: enumerate devices, connect, receive
//! note events through a callback on a background thread.

pub mod buffer;
pub mod capture;
pub mod detector;
pub mod manager;
pub mod model_store;
pub mod pipeline;
pub mod resample;
pub mod tracker;

pub use capture::MicDevice;
pub use manager::{AudioInputConnection, AudioInputError, AudioInputManager};
pub use tracker::MicEvent;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Model frame stride: 16kHz / 100fps.
pub const SAMPLES_PER_FRAME: usize = 160;
/// Streaming inference window: 1.5s.
pub const WINDOW_SAMPLES: usize = 24_000;
/// Window hop: 60ms.
pub const HOP_SAMPLES: usize = 3_200;
/// Frames near the window edge whose predictions are unreliable.
pub const TRUST_MARGIN_FRAMES: usize = 12;
