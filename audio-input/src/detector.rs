//! Streaming inference seam: frame-probability types and the pluggable
//! `PitchDetector` trait (design §5). Deliberately free of backend types
//! so the pipeline can be tested with mocks.

pub const KEY_COUNT: usize = 88;
pub const FIRST_MIDI_KEY: u8 = 21;

/// Per-window frame-level model output, row-major [frame][pitch].
/// `first_frame` is the global frame index of row 0.
#[derive(Debug, Clone)]
pub struct FrameProbabilities {
    pub first_frame: usize,
    pub frames: usize,
    /// onset posterior, already thresholded by the detector
    pub onset: Vec<bool>,
    /// frame activation posterior, raw f32 in [0,1]
    pub frame: Vec<f32>,
}

impl FrameProbabilities {
    #[inline]
    pub fn onset_at(&self, frame: usize, pitch: usize) -> bool {
        self.onset[frame * KEY_COUNT + pitch]
    }

    #[inline]
    pub fn frame_at(&self, frame: usize, pitch: usize) -> f32 {
        self.frame[frame * KEY_COUNT + pitch]
    }
}

/// Pluggable inference backend seam (design §5).
pub trait PitchDetector: Send {
    /// `window`: mono f32 @16kHz, the pipeline window length (see lib consts).
    /// `first_frame`: global frame index of window start.
    ///
    /// Panic contract: implementations may panic on inference failure
    /// (bad model, shape mismatch). Callers must contain panics — the
    /// inference thread wraps detect() in catch_unwind and converts a
    /// panic into a MicEvent::Error.
    fn detect(&mut self, window: &[f32], first_frame: usize) -> FrameProbabilities;
}

/// Deterministic detector for tests: replays scripted windows.
pub struct MockDetector {
    /// Queue of results; one popped per detect() call.
    pub scripted: std::collections::VecDeque<FrameProbabilities>,
}

impl PitchDetector for MockDetector {
    fn detect(&mut self, _window: &[f32], first_frame: usize) -> FrameProbabilities {
        let mut r = self
            .scripted
            .pop_front()
            .expect("MockDetector ran out of scripted results");
        r.first_frame = first_frame;
        r
    }
}

/// Helper to build a silent window result of `frames` frames.
pub fn silent_frames(first_frame: usize, frames: usize) -> FrameProbabilities {
    FrameProbabilities {
        first_frame,
        frames,
        onset: vec![false; frames * KEY_COUNT],
        frame: vec![0.0; frames * KEY_COUNT],
    }
}

/// rten (ONNX-derived) inference backend (design §5).
pub mod rten_backend {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;

    use super::{FrameProbabilities, KEY_COUNT, PitchDetector};
    use neothesia_ai::ONSET_THRESHOLD;

    pub struct RtenDetector {
        model: rten::Model,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("failed to load pitch detection model: {0}")]
    pub struct DetectError(String);

    impl RtenDetector {
        pub fn load(model_path: &std::path::Path) -> Result<Self, DetectError> {
            let model =
                rten::Model::load_file(model_path).map_err(|e| DetectError(e.to_string()))?;

            // Validate the model graph up front so a wrong file fails
            // with a clean load error instead of a first-detect panic.
            if model.input_ids().len() != 1 {
                return Err(DetectError(format!(
                    "expected model with 1 input, found {}",
                    model.input_ids().len()
                )));
            }
            if model.output_ids().len() != 7 {
                return Err(DetectError(format!(
                    "expected model with 7 outputs, found {}",
                    model.output_ids().len()
                )));
            }

            Ok(Self { model })
        }
    }

    impl PitchDetector for RtenDetector {
        fn detect(&mut self, window: &[f32], first_frame: usize) -> FrameProbabilities {
            assert!(
                window.len() % crate::SAMPLES_PER_FRAME == 0,
                "window length {} is not a multiple of SAMPLES_PER_FRAME",
                window.len()
            );
            let frames = window.len() / crate::SAMPLES_PER_FRAME;

            let input = Tensor::from_data(&[1, window.len()], window.to_vec());

            let inputs: Vec<(rten::NodeId, rten::ValueOrView)> =
                vec![(self.model.input_ids()[0], input.view().into())];

            let outputs = self
                .model
                .run_n::<7>(inputs, self.model.output_ids().try_into().unwrap(), None)
                .expect("model inference failed");

            // Output order matches the neothesia-ai offline CLI:
            // [reg_onset, reg_offset, frame, velocity, pedal_onset, pedal_offset, pedal_frame]
            let [reg_onset, _reg_offset, frame, ..] = outputs;

            let onset_tensor = reg_onset.into_tensor::<f32>().unwrap();
            let frame_tensor = frame.into_tensor::<f32>().unwrap();

            // Model layout: [1, frames, 88] — same framing as the offline path
            let onset_flat = onset_tensor.to_vec();
            let frame_flat = frame_tensor.to_vec();

            assert_eq!(onset_flat.len(), frames * KEY_COUNT);
            assert_eq!(frame_flat.len(), frames * KEY_COUNT);

            let onset = onset_flat.iter().map(|&p| p > ONSET_THRESHOLD).collect();

            FrameProbabilities {
                first_frame,
                frames,
                onset,
                frame: frame_flat,
            }
        }
    }
}

#[cfg(test)]
mod rten_tests {
    use super::rten_backend::RtenDetector;
    use super::*;
    use crate::{SAMPLES_PER_FRAME, WINDOW_SAMPLES};

    fn model_path() -> Option<std::path::PathBuf> {
        std::env::var("NTS_TEST_MODEL").ok().map(Into::into)
    }

    #[test]
    fn load_nonexistent_path_fails() {
        let err = RtenDetector::load(std::path::Path::new(
            "/nonexistent/definitely-no-model-here.rten",
        ))
        .expect_err("load must fail for a nonexistent path");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn load_corrupt_file_fails() {
        let dir = std::env::temp_dir().join("nts-detector-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("corrupt.rten");
        std::fs::write(&f, b"this is not an rten model").unwrap();
        assert!(RtenDetector::load(&f).is_err());
    }

    /// A silent window must not produce any onset.
    #[test]
    #[ignore = "requires NTS_TEST_MODEL=<path to .rten model>"]
    fn silence_produces_no_onsets() {
        let Some(path) = model_path() else {
            eprintln!("skipping: NTS_TEST_MODEL not set");
            return;
        };
        let mut det = RtenDetector::load(&path).unwrap();
        let window = vec![0.0_f32; WINDOW_SAMPLES];
        let probs = det.detect(&window, 0);
        assert!(probs.onset.iter().all(|&o| !o));
        assert_eq!(probs.frames, WINDOW_SAMPLES / SAMPLES_PER_FRAME);
    }
}
