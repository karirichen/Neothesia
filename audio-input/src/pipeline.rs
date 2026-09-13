//! Streaming inference pipeline (design §5): chains the rolling history
//! buffer, per-hop window, RMS gate, detector, and note tracker.
//!
//! Each 60ms hop slides the 1.5s window by `HOP_SAMPLES`, runs detection,
//! and feeds only the newly-trusted frames (those at least
//! `TRUST_MARGIN_FRAMES` from the window's right edge) to the tracker as
//! a monotonic frontier — a frame is never re-fed.

use std::collections::VecDeque;

use crate::detector::{FrameProbabilities, PitchDetector};
use crate::tracker::{MicEvent, NoteTracker, TrackerConfig};
use crate::{HOP_SAMPLES, SAMPLES_PER_FRAME, TRUST_MARGIN_FRAMES, WINDOW_SAMPLES};

/// RMS gate: windows below this are skipped entirely (CPU saver).
pub const RMS_GATE: f32 = 0.001;

pub struct StreamingPipeline<D: PitchDetector> {
    detector: D,
    tracker: NoteTracker,
    /// Rolling 16kHz mono history, len <= WINDOW_SAMPLES.
    history: VecDeque<f32>,
    /// 16kHz samples consumed since connect.
    total_samples: usize,
    /// 16kHz samples consumed at last inference run.
    last_run_sample: usize,
    /// Global frame index one past the last frame fed to the tracker.
    /// Trusted ranges are fed monotonically from here — never re-feed a
    /// frame, never start a range before the current window (the tracker
    /// asserts monotonicity; overlapping re-feeds underflow its frame
    /// arithmetic).
    last_trusted_end: usize,
}

impl<D: PitchDetector> StreamingPipeline<D> {
    pub fn new(detector: D, tracker_cfg: TrackerConfig) -> Self {
        Self {
            detector,
            tracker: NoteTracker::new(tracker_cfg),
            history: VecDeque::with_capacity(WINDOW_SAMPLES),
            total_samples: 0,
            last_run_sample: 0,
            last_trusted_end: 0,
        }
    }

    /// Feed newly resampled 16kHz mono samples; returns events.
    pub fn push(&mut self, samples: &[f32]) -> Vec<MicEvent> {
        self.total_samples += samples.len();
        for &s in samples {
            if self.history.len() == WINDOW_SAMPLES {
                self.history.pop_front();
            }
            self.history.push_back(s);
        }

        let mut events = Vec::new();

        while self.total_samples >= WINDOW_SAMPLES
            && self.total_samples - self.last_run_sample >= HOP_SAMPLES
        {
            self.last_run_sample = self.total_samples;
            events.extend(self.run_once());
        }

        events
    }

    fn run_once(&mut self) -> Vec<MicEvent> {
        let window: Vec<f32> = self.history.iter().copied().collect();
        debug_assert_eq!(window.len(), WINDOW_SAMPLES);

        // energy gate
        let rms = (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt();
        if rms < RMS_GATE {
            // prolonged silence: early-release sustained notes is a
            // possible optimization; conservatively just skip
            return Vec::new();
        }

        let window_start_sample = self.total_samples - WINDOW_SAMPLES;
        let first_frame = window_start_sample / SAMPLES_PER_FRAME;
        let probs: FrameProbabilities = self.detector.detect(&window, first_frame);

        // Feed only NEW trusted frames: a monotonic frontier
        // [last_trusted_end, new_trusted_end) clamped to this window.
        let window_frames = WINDOW_SAMPLES / SAMPLES_PER_FRAME;
        let new_trusted_end = first_frame + window_frames - TRUST_MARGIN_FRAMES;
        let feed_start = self.last_trusted_end.max(first_frame);
        if new_trusted_end > feed_start {
            self.last_trusted_end = new_trusted_end;
            self.tracker.process(feed_start, new_trusted_end, &probs)
        } else {
            Vec::new()
        }
    }

    pub fn all_notes_off(&mut self) -> Vec<MicEvent> {
        self.tracker.all_notes_off()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{MockDetector, silent_frames};

    fn mk_pipeline(scripted: Vec<FrameProbabilities>) -> StreamingPipeline<MockDetector> {
        StreamingPipeline::new(
            MockDetector {
                scripted: scripted.into(),
            },
            TrackerConfig::default(),
        )
    }

    /// Feed enough audio and verify the trusted range maps to the
    /// correct global frame indices.
    #[test]
    fn trust_boundary_maps_to_global_frames() {
        // window 1: first full window, first_frame=0, onset at global
        // frame 3 (trusted). Activation is sustained from the onset to
        // the end of the window — a struck-and-held note — otherwise the
        // tracker's release logic would emit a spurious NoteOff inside
        // the fed range and muddy the frame-mapping assertion.
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[3 * 88 + 40] = true;
            for f in 3..150 {
                pr.frame[f * 88 + 40] = 0.9;
            }
            pr
        };
        let mut p = mk_pipeline(vec![w1]);

        // the RMS gate would block all-zero windows → feed non-zero noise floor
        let noise = vec![0.01_f32; WINDOW_SAMPLES];
        let ev = p.push(&noise);

        assert_eq!(ev, vec![MicEvent::NoteOn { key: 21 + 40 }]);
    }

    #[test]
    fn untrusted_margin_frames_are_deferred() {
        // onset appears at global frame 145 (>= 150-12=138) → must not
        // trigger in window 1. Each hop advances the window start by
        // HOP_SAMPLES=960 → first_frame by 6 frames, so:
        //   w1: first_frame=0,  onset local 145; trusted feed [0,138)
        //   w2: first_frame=6,  onset local 139; trusted feed [138,144)
        //   w3: first_frame=12, onset local 133; trusted feed [144,150)
        // The onset first becomes trusted in window 3 (local 133 ∈ [132,138)).
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[145 * 88 + 40] = true;
            pr.frame[145 * 88 + 40] = 0.9;
            pr
        };
        let w2 = {
            let mut pr = silent_frames(6, 150);
            pr.onset[139 * 88 + 40] = true;
            pr.frame[139 * 88 + 40] = 0.9;
            pr
        };
        let w3 = {
            let mut pr = silent_frames(12, 150);
            pr.onset[133 * 88 + 40] = true;
            pr.frame[133 * 88 + 40] = 0.9;
            pr
        };
        let mut p = mk_pipeline(vec![w1, w2, w3]);

        let noise = vec![0.01_f32; WINDOW_SAMPLES];
        assert!(
            p.push(&noise).is_empty(),
            "margin frames must not trigger early"
        );

        let noise2 = vec![0.01_f32; HOP_SAMPLES];
        assert!(p.push(&noise2).is_empty(), "still untrusted in window 2");

        let noise3 = vec![0.01_f32; HOP_SAMPLES];
        let ev = p.push(&noise3);
        assert_eq!(ev, vec![MicEvent::NoteOn { key: 21 + 40 }]);
    }

    #[test]
    fn rms_gate_skips_silence() {
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[3 * 88 + 40] = true;
            pr
        };
        let mut p = mk_pipeline(vec![w1]);
        let silence = vec![0.0_f32; WINDOW_SAMPLES];
        assert!(p.push(&silence).is_empty());
        assert_eq!(p.history.len(), WINDOW_SAMPLES);
    }
}
