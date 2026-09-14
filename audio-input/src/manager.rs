//! Public API of the crate (design §5): enumerate mic devices,
//! connect, and receive note events on a background thread —
//! mirroring `midi_io`'s shape.

use std::sync::Arc;
use std::time::Duration;

use crate::buffer::SampleBuffer;
use crate::capture::{self, MicDevice};
use crate::detector::rten_backend::RtenDetector;
use crate::pipeline::StreamingPipeline;
use crate::tracker::{MicEvent, TrackerConfig};

#[derive(Debug, thiserror::Error)]
pub enum AudioInputError {
    #[error(transparent)]
    Capture(#[from] capture::CaptureError),
    #[error("pitch detector failed to load: {0}")]
    Detector(String),
}

pub struct AudioInputConnection {
    pub device: MicDevice,
    /// Set to true on drop; the inference thread exits its loop.
    stop: Arc<std::sync::atomic::AtomicBool>,
    _stream: capture::CaptureStream,
}

impl Drop for AudioInputConnection {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

pub struct AudioInputManager;

impl AudioInputManager {
    pub fn devices() -> Vec<MicDevice> {
        capture::devices()
    }

    /// Heuristic default device when the user has not picked one:
    /// prefer a built-in mic over Continuity/virtual devices (macOS
    /// enumeration often puts "…iPhone… Microphone" first).
    pub fn default_device() -> Option<MicDevice> {
        pick_default_device(&capture::devices()).cloned()
    }

    /// Connect to `device`, run the streaming pipeline on a background
    /// thread, deliver events via `on_event`. `model_path` must point
    /// to a valid .rten model file.
    pub fn connect<F>(
        device: &MicDevice,
        model_path: &std::path::Path,
        mut on_event: F,
    ) -> Result<AudioInputConnection, AudioInputError>
    where
        F: FnMut(MicEvent) + Send + 'static,
    {
        let stream = capture::connect(device)?;
        let detector =
            RtenDetector::load(model_path).map_err(|e| AudioInputError::Detector(e.to_string()))?;

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop2 = stop.clone();

        let buffer: Arc<SampleBuffer> = stream.buffer.clone();
        let source_rate = stream.sample_rate;
        let capture_error = stream.error.clone();

        std::thread::Builder::new()
            .name("audio-input-inference".into())
            .spawn(move || {
                let mut resampler = crate::resample::ResampleStage::new(source_rate);
                let mut pipeline = StreamingPipeline::new(detector, TrackerConfig::default());
                let (tx, rx) = std::sync::mpsc::channel::<MicEvent>();

                // Event relay: the user callback runs off the inference
                // thread. After the connection is dropped, the inference
                // thread may still deliver events for up to ~60ms (one
                // poll cycle) before `tx` drops — consumers must treat
                // late events as harmless (e.g. EventLoopProxy send errors
                // are ignorable).
                std::thread::Builder::new()
                    .name("audio-input-relay".into())
                    .spawn(move || {
                        while let Ok(ev) = rx.recv() {
                            on_event(ev);
                        }
                    })
                    .expect("failed to spawn audio-input relay thread");

                while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(20));

                    // device-level error (hot-unplug etc.): notify +
                    // force-release all notes + exit
                    if capture_error.load(std::sync::atomic::Ordering::Relaxed) {
                        for ev in pipeline.all_notes_off() {
                            let _ = tx.send(ev);
                        }
                        let _ = tx.send(MicEvent::Error("input stream error (device lost?)"));
                        break;
                    }

                    let raw = buffer.drain();
                    if raw.is_empty() {
                        continue;
                    }

                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let resampled = resampler.process(&raw);
                        pipeline.push(&resampled)
                    }));

                    match result {
                        Ok(events) => {
                            for ev in events {
                                let _ = tx.send(ev);
                            }
                        }
                        Err(payload) => {
                            // Downcast the panic payload for diagnosability
                            // (e.g. model shape mismatches surface here).
                            let msg = payload
                                .downcast_ref::<&str>()
                                .map(|s| (*s).to_string())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown panic".to_string());
                            log::error!("audio-input inference panicked: {msg}");
                            let _ = tx.send(MicEvent::Error("inference thread panicked"));
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                for ev in pipeline.all_notes_off() {
                                    let _ = tx.send(ev);
                                }
                            }));
                            break;
                        }
                    }
                }

                // Clean stop (connection dropped): release anything
                // still sounding so no key stays highlighted forever.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for ev in pipeline.all_notes_off() {
                        let _ = tx.send(ev);
                    }
                }));
            })
            .expect("failed to spawn audio-input thread");

        Ok(AudioInputConnection {
            device: device.clone(),
            stop,
            _stream: stream,
        })
    }
}

/// Pure ranking used by [`AudioInputManager::default_device`]:
/// built-in-style mics first (e.g. "MacBook Pro Microphone"), then
/// anything that is not a phone/Continuity/virtual device, then
/// whatever is first.
fn pick_default_device(devices: &[MicDevice]) -> Option<&MicDevice> {
    let looks_builtin = |n: &str| {
        (n.contains("MacBook") || n.contains("Microphone") || n.contains("Internal"))
            && !n.contains("iPhone")
            && !n.contains("iPad")
    };
    let looks_virtual = |n: &str| {
        n.contains("iPhone")
            || n.contains("iPad")
            || n.contains("Teams")
            || n.contains("Aggregate")
            || n.contains("Virtual")
    };

    devices
        .iter()
        .find(|d| looks_builtin(&d.0))
        .or_else(|| devices.iter().find(|d| !looks_virtual(&d.0)))
        .or_else(|| devices.first())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mic(name: &str) -> MicDevice {
        MicDevice(name.to_string())
    }

    /// Real-world macOS enumeration order: Continuity iPhone first.
    #[test]
    fn default_device_prefers_builtin_over_continuity() {
        let devices = vec![
            mic("Karina's iPhone 12 Microphone"),
            mic("MacBook Pro Microphone"),
            mic("Microsoft Teams Audio"),
        ];
        assert_eq!(
            pick_default_device(&devices).map(|d| d.0.as_str()),
            Some("MacBook Pro Microphone")
        );
    }

    #[test]
    fn default_device_falls_back_to_first_non_virtual() {
        let devices = vec![mic("Some Virtual Driver"), mic("Yeti USB Mic")];
        assert_eq!(
            pick_default_device(&devices).map(|d| d.0.as_str()),
            Some("Yeti USB Mic")
        );
    }

    #[test]
    fn default_device_last_resort_is_first() {
        let devices = vec![mic("Only Virtual")];
        assert_eq!(
            pick_default_device(&devices).map(|d| d.0.as_str()),
            Some("Only Virtual")
        );
    }
}
