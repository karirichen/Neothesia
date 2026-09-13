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
