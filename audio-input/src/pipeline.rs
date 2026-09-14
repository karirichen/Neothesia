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
    /// RMS of the most recent window (diagnostic heartbeat).
    pub last_rms: f32,
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
            last_rms: 0.0,
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

        // NOTE: this is effectively a single `if` — `last_run_sample`
        // is snapped to `total_samples`, so the loop condition is false
        // after one pass. Chunks from the 20ms poll loop are far below
        // the hop size; a push >= 2x HOP runs inference once, which is
        // fine for realtime capture.
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

        // Window start is push-boundary aligned, not 160-sample aligned,
        // so `first_frame` truncation carries up to ~10ms phase error vs
        // the global grid — well inside tracker tolerance; do not "fix".
        let window_start_sample = self.total_samples - WINDOW_SAMPLES;
        let first_frame = window_start_sample / SAMPLES_PER_FRAME;

        // energy gate
        let rms = (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt();
        self.last_rms = rms;
        if rms < RMS_GATE {
            // Silence has outlasted the trust margin when the window
            // start has slid past the fed frontier: any sounding notes
            // ended (the tracker's frame clock is frozen while unfed,
            // so its release/lifetime logic would never fire). Release
            // them and retire the silent gap.
            if first_frame > self.last_trusted_end {
                self.last_trusted_end = first_frame;
                return self.tracker.all_notes_off();
            }
            return Vec::new();
        }

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
        // HOP_SAMPLES=3200 → first_frame by 20 frames, so:
        //   w1: first_frame=0,  onset local 145; trusted feed [0,138)
        //   w2: first_frame=20, onset local 125; trusted feed [138,158)
        // The onset first becomes trusted in window 2 (local 125 < 138).
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[145 * 88 + 40] = true;
            pr.frame[145 * 88 + 40] = 0.9;
            pr
        };
        let w2 = {
            let mut pr = silent_frames(20, 150);
            pr.onset[125 * 88 + 40] = true;
            pr.frame[125 * 88 + 40] = 0.9;
            pr
        };
        let mut p = mk_pipeline(vec![w1, w2]);

        let noise = vec![0.01_f32; WINDOW_SAMPLES];
        assert!(
            p.push(&noise).is_empty(),
            "margin frames must not trigger early"
        );

        let noise2 = vec![0.01_f32; HOP_SAMPLES];
        let ev = p.push(&noise2);
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

    /// A sounding note must be force-released once the window has slid
    /// fully into silence, and later sound must still detect — the
    /// tracker's frozen frame clock must not wedge the pipeline.
    /// Silence/resume are pushed in full-window chunks so each phase is
    /// exactly one run (mixed windows would pass the RMS gate and
    /// consume extra scripted detections).
    #[test]
    fn sustained_silence_releases_then_recovers() {
        // Window 1 (first_frame=0): onset + sustained activation → NoteOn
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[3 * 88 + 40] = true;
            for f in 3..150 {
                pr.frame[f * 88 + 40] = 0.9;
            }
            pr
        };
        // Resume window: 3 full-window pushes later the window start is
        // (96000-24000)/160 = frame 450; onset at global 460 → local 10.
        let w_resume = {
            let mut pr = silent_frames(450, 150);
            pr.onset[10 * 88 + 40] = true;
            for f in 10..150 {
                pr.frame[f * 88 + 40] = 0.9;
            }
            pr
        };
        let mut p = mk_pipeline(vec![w1, w_resume]);

        let ev = p.push(&vec![0.01_f32; WINDOW_SAMPLES]);
        assert_eq!(ev, vec![MicEvent::NoteOn { key: 21 + 40 }]);

        // First fully-silent window: gate trips, first_frame(150) has
        // slid past the frontier(138) → force-release.
        let ev2 = p.push(&vec![0.0_f32; WINDOW_SAMPLES]);
        assert_eq!(ev2, vec![MicEvent::NoteOff { key: 21 + 40 }]);

        // Continued silence: nothing sounding, nothing detected.
        let ev3 = p.push(&vec![0.0_f32; WINDOW_SAMPLES]);
        assert!(ev3.is_empty());

        // Sound resumes: fresh onset must still fire (frozen-clock
        // wedge regression).
        let ev4 = p.push(&vec![0.01_f32; WINDOW_SAMPLES]);
        assert_eq!(ev4, vec![MicEvent::NoteOn { key: 21 + 40 }]);
    }

    /// A single push larger than the hop size runs inference once and
    /// keeps the frontier gap-free (pins the single-run semantics).
    #[test]
    fn multi_hop_burst_runs_once_and_stays_monotonic() {
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[3 * 88 + 40] = true;
            for f in 3..150 {
                pr.frame[f * 88 + 40] = 0.9;
            }
            pr
        };
        // The burst run's window: one 5*HOP(=16000)-sample push slides
        // the window start to 16000 → first_frame = 100. Feed
        // [138, 238): 100 silent frames, so the held note's release
        // window (20 frames) elapses mid-range → exactly one NoteOff.
        let w2 = silent_frames(100, 150);
        let mut p = mk_pipeline(vec![w1, w2]);

        let ev = p.push(&vec![0.01_f32; WINDOW_SAMPLES]);
        assert_eq!(ev, vec![MicEvent::NoteOn { key: 21 + 40 }]);

        // A 5-hop burst triggers exactly one run — the single scripted
        // window is consumed and none remain.
        let ev2 = p.push(&vec![0.01_f32; 5 * HOP_SAMPLES]);
        assert_eq!(ev2, vec![MicEvent::NoteOff { key: 21 + 40 }]);
        assert!(
            p.detector.scripted.is_empty(),
            "burst must consume exactly one scripted window"
        );
    }
}
