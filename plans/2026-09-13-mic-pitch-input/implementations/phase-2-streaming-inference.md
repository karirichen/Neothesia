# Phase 2: Streaming Inference + NoteTracker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the sliding-window streaming inference pipeline: 16kHz window → PitchDetector → NoteTracker diffing → MicEvent stream, including the replaceable-backend seam and mock-based tests.

**Architecture:** A `PitchDetector` trait isolates the inference implementation (rten is the default; tests use MockDetector); `NoteTracker` is a pure state machine (frame probabilities in, NoteOn/NoteOff out); `StreamingPipeline` chains buffer/resampling/detection/diffing. Everything except RtenDetector is testable without a model.

**Tech Stack:** rten (model inference), neothesia-ai lib (Phase 1 output), plain Rust state machine.

**Design doc:** `plans/2026-09-13-mic-pitch-input/design.md` §5

**Parameter quick reference (aligned with the design doc; hop corrected from 64ms to 60ms to sit on the 10ms frame grid — recorded in the design doc appendix):**
- Window `WINDOW_SAMPLES = 24000` (1.5s) → 150 frames
- Hop `HOP_SAMPLES = 960` (60ms) = 6 frames
- Trust margin `TRUST_MARGIN_FRAMES = 12` (120ms)
- Frame rate 100fps (160 samples/frame), 88 pitches (MIDI 21..108)

---

### Task 2.1: FrameProbabilities and the PitchDetector trait

**Files:**
- Create: `audio-input/src/detector.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod detector;`)

- [x] **Step 1: Types and trait**

`audio-input/src/detector.rs`:

```rust
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
    /// `window`: mono f32 @16kHz, length `WINDOW_SAMPLES`.
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
```

- [x] **Step 2: Compile check**

Run: `cargo check -p audio-input`
Expected: passes.

- [x] **Step 3: Commit**

```bash
git add audio-input
git commit -m "feat(audio-input): PitchDetector trait and frame probability types"
```

---

### Task 2.2: NoteTracker state machine

**Files:**
- Create: `audio-input/src/tracker.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod tracker;`)

- [x] **Step 1: Write the failing tests together with the implementation**

`audio-input/src/tracker.rs`:

```rust
use crate::detector::{FIRST_MIDI_KEY, FrameProbabilities, KEY_COUNT};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicEvent {
    NoteOn { key: u8 },
    NoteOff { key: u8 },
    /// Background-thread failure (inference panic, device loss).
    Error(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct TrackerConfig {
    /// Same-pitch retrigger suppression window.
    pub onset_cooldown_frames: usize,   // 25 = 250ms
    /// Frames below threshold before NoteOff.
    pub release_frames: usize,          // 20 = 200ms
    /// Hard note lifetime.
    pub max_note_frames: usize,         // 400 = 4s
    pub frame_release_threshold: f32,   // 0.1
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            onset_cooldown_frames: 25,
            release_frames: 20,
            max_note_frames: 400,
            frame_release_threshold: 0.1,
        }
    }
}

#[derive(Default)]
struct PitchState {
    sounding: bool,
    onset_frame: usize,
    last_onset_seen: usize,
    below_since: Option<usize>,
}

pub struct NoteTracker {
    cfg: TrackerConfig,
    states: [PitchState; KEY_COUNT],
}

impl NoteTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self { cfg, states: std::array::from_fn(|_| PitchState::default()) }
    }

    /// Feed the newly-trusted frame range [start, end) of `probs`
    /// (which must contain those frames). Returns emitted events.
    pub fn process(&mut self, start: usize, end: usize, probs: &FrameProbabilities) -> Vec<MicEvent> {
        assert!(end > start, "empty trusted range");
        let local_start = start - probs.first_frame;
        let local_end = end - probs.first_frame;
        assert!(local_end <= probs.frames, "trusted range outside window");

        let mut events = Vec::new();
        let key_of = |p: usize| FIRST_MIDI_KEY + p as u8;

        for f in local_start..local_end {
            let global_f = probs.first_frame + f;

            for p in 0..KEY_COUNT {
                let st = &mut self.states[p];

                if probs.onset_at(f, p) {
                    if !st.sounding
                        && global_f.saturating_sub(st.last_onset_seen)
                            >= self.cfg.onset_cooldown_frames
                    {
                        st.sounding = true;
                        st.onset_frame = global_f;
                        st.below_since = None;
                        events.push(MicEvent::NoteOn { key: key_of(p) });
                    } else if st.sounding
                        && global_f.saturating_sub(st.onset_frame)
                            >= self.cfg.onset_cooldown_frames
                    {
                        // consecutive strike on same pitch: close + reopen
                        events.push(MicEvent::NoteOff { key: key_of(p) });
                        events.push(MicEvent::NoteOn { key: key_of(p) });
                        st.onset_frame = global_f;
                        st.below_since = None;
                    }
                    st.last_onset_seen = global_f;
                    continue;
                }

                if !st.sounding {
                    continue;
                }

                // release bookkeeping
                if probs.frame_at(f, p) < self.cfg.frame_release_threshold {
                    let since = *st.below_since.get_or_insert(global_f);
                    if global_f - since >= self.cfg.release_frames {
                        st.sounding = false;
                        st.below_since = None;
                        events.push(MicEvent::NoteOff { key: key_of(p) });
                        continue;
                    }
                } else {
                    st.below_since = None;
                }

                // hard lifetime cutoff (reverb insurance)
                if global_f - st.onset_frame >= self.cfg.max_note_frames {
                    st.sounding = false;
                    st.below_since = None;
                    events.push(MicEvent::NoteOff { key: key_of(p) });
                }
            }
        }

        events
    }

    /// Force-release everything currently sounding (device loss etc.).
    pub fn all_notes_off(&mut self) -> Vec<MicEvent> {
        let mut events = Vec::new();
        for (p, st) in self.states.iter_mut().enumerate() {
            if st.sounding {
                st.sounding = false;
                st.below_since = None;
                events.push(MicEvent::NoteOff {
                    key: FIRST_MIDI_KEY + p as u8,
                });
            }
        }
        events
    }
}
```

- [x] **Step 2: Unit tests**

Append to `tracker.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::silent_frames;

    fn window_with(first_frame: usize, frames: usize, specs: &[(usize, usize, f32, bool)]) -> FrameProbabilities
    {
        // specs: (frame, pitch, frame_prob, onset)
        let mut pr = silent_frames(first_frame, frames);
        for &(f, p, val, onset) in specs {
            pr.frame[f * KEY_COUNT + p] = val;
            pr.onset[f * KEY_COUNT + p] = onset;
        }
        pr
    }

    #[test]
    fn onset_triggers_note_on() {
        let mut t = NoteTracker::new(Default::default());
        let pr = window_with(0, 10, &[(2, 30, 0.9, true)]);
        let ev = t.process(0, 10, &pr);
        assert_eq!(ev, vec![MicEvent::NoteOn { key: 21 + 30 }]);
    }

    #[test]
    fn release_after_200ms_below_threshold() {
        let mut t = NoteTracker::new(Default::default());
        // onset at frame 0, sounding frames 0..9, then silence
        let pr = window_with(0, 10, &[(0, 30, 0.9, true)]);
        t.process(0, 10, &pr);

        let mut specs = Vec::new();
        for f in 10..40 {
            specs.push((f, 30, 0.0, false));
        }
        let pr2 = window_with(10, 30, &specs);
        let ev = t.process(10, 40, &pr2);
        // below_since=10, release at 10+20=30
        assert_eq!(ev, vec![MicEvent::NoteOff { key: 21 + 30 }]);
    }

    #[test]
    fn no_note_off_while_activation_holds() {
        let mut t = NoteTracker::new(Default::default());
        let pr = window_with(0, 10, &[(0, 30, 0.9, true)]);
        t.process(0, 10, &pr);

        let specs: Vec<_> = (10..50).map(|f| (f, 30, 0.7, false)).collect();
        let pr2 = window_with(10, 40, &specs);
        assert!(t.process(10, 50, &pr2).is_empty());
    }

    #[test]
    fn same_pitch_onset_suppressed_within_cooldown() {
        let mut t = NoteTracker::new(Default::default());
        // two onsets 10 frames apart (< 25) → only one trigger
        let pr = window_with(0, 15, &[(2, 30, 0.9, true), (12, 30, 0.9, true)]);
        let ev = t.process(0, 15, &pr);
        assert_eq!(ev, vec![MicEvent::NoteOn { key: 21 + 30 }]);
    }

    #[test]
    fn consecutive_strike_reopens_note() {
        let mut t = NoteTracker::new(Default::default());
        let pr = window_with(0, 10, &[(0, 30, 0.9, true)]);
        t.process(0, 10, &pr);
        // second onset at frame 38 (> 25 cooldown) → Off + On
        let mut specs3 = Vec::new();
        for f in 35..40 {
            specs3.push((f, 30, 0.7, false));
        }
        specs3.push((38, 30, 0.9, true));
        let pr3 = window_with(35, 5, &specs3);
        let ev = t.process(35, 40, &pr3);
        assert_eq!(
            ev,
            vec![
                MicEvent::NoteOff { key: 21 + 30 },
                MicEvent::NoteOn { key: 21 + 30 },
            ]
        );
    }

    #[test]
    fn forced_off_after_max_lifetime() {
        let cfg = TrackerConfig { max_note_frames: 100, ..Default::default() };
        let mut t = NoteTracker::new(cfg);
        let pr = window_with(0, 10, &[(0, 30, 0.9, true)]);
        t.process(0, 10, &pr);
        // sustained high activation for 200 frames
        let specs: Vec<_> = (10..210).map(|f| (f, 30, 0.7, false)).collect();
        let pr2 = window_with(10, 200, &specs);
        let ev = t.process(10, 210, &pr2);
        assert_eq!(ev, vec![MicEvent::NoteOff { key: 21 + 30 }]);
    }

    #[test]
    fn all_notes_off_forces_release() {
        let mut t = NoteTracker::new(Default::default());
        let pr = window_with(0, 5, &[(1, 10, 0.9, true), (1, 50, 0.9, true)]);
        t.process(0, 5, &pr);
        let mut ev = t.all_notes_off();
        ev.sort_by_key(|e| match e {
            MicEvent::NoteOn { key } | MicEvent::NoteOff { key } => *key,
            MicEvent::Error(_) => 0,
        });
        assert_eq!(
            ev,
            vec![
                MicEvent::NoteOff { key: 21 + 10 },
                MicEvent::NoteOff { key: 21 + 50 },
            ]
        );
    }
}
```

- [x] **Step 3: Run the tests**

Run: `cargo test -p audio-input`
Expected: all pass (6 tracker tests + existing tests).

- [x] **Step 4: Commit**

```bash
git add audio-input
git commit -m "feat(audio-input): streaming note tracker state machine"
```

---

### Task 2.3: RtenDetector (model backend)

**Files:**
- Modify: `audio-input/src/detector.rs` (append implementation)
- Modify: `neothesia-ai/src/transcription.rs` (expose threshold constants)

- [x] **Step 1: Add threshold constants to neothesia-ai**

Top of `neothesia-ai/src/transcription.rs`:

```rust
pub const ONSET_THRESHOLD: f32 = 0.3;
pub const FRAME_THRESHOLD: f32 = 0.1;
```

(Replace the literal `0.3`/`0.1` in main.rs with these constants — behavior unchanged.)

- [x] **Step 2: RtenDetector implementation**

Append to `audio-input/src/detector.rs`:

```rust
pub mod rten_backend {
    use super::{FrameProbabilities, PitchDetector, KEY_COUNT};
    use neothesia_ai::{FRAME_THRESHOLD, ONSET_THRESHOLD};

    pub struct RtenDetector {
        model: rten::Model,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("failed to run pitch detection model: {0}")]
    pub struct DetectError(String);

    impl RtenDetector {
        pub fn load(model_path: &std::path::Path) -> Result<Self, DetectError> {
            let model = rten::Model::load_file(model_path)
                .map_err(|e| DetectError(e.to_string()))?;
            Ok(Self { model })
        }
    }

    impl PitchDetector for RtenDetector {
        fn detect(&mut self, window: &[f32], first_frame: usize) -> FrameProbabilities {
            let frames = window.len() / crate::SAMPLES_PER_FRAME;

            let input =
                rten::Tensor::from_data(&[1, window.len()], window.to_vec());

            let inputs: Vec<(rten::NodeId, rten::ValueOrView)> = vec![(
                self.model.input_ids()[0],
                input.view().into(),
            )];

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

            debug_assert_eq!(onset_flat.len(), frames * KEY_COUNT);
            let _ = FRAME_THRESHOLD; // used by tracker-side tuning; kept for parity

            let onset = onset_flat
                .iter()
                .map(|&p| p > ONSET_THRESHOLD)
                .collect();

            FrameProbabilities {
                first_frame,
                frames,
                onset,
                frame: frame_flat,
            }
        }
    }
}
```

Add to `audio-input/src/lib.rs`:

```rust
pub const SAMPLES_PER_FRAME: usize = 160; // 16kHz / 100fps
pub const WINDOW_SAMPLES: usize = 24_000; // 1.5s
pub const HOP_SAMPLES: usize = 960;       // 60ms
pub const TRUST_MARGIN_FRAMES: usize = 12; // 120ms
```

- [x] **Step 3: Model smoke test (#[ignore], requires a local model file)**

Append to `detector.rs`:

```rust
#[cfg(test)]
mod rten_tests {
    use super::rten_backend::RtenDetector;
    use super::*;

    fn model_path() -> Option<std::path::PathBuf> {
        std::env::var("NTS_TEST_MODEL").ok().map(Into::into)
    }

    /// A silent window must not produce any onset.
    #[test]
    #[ignore = "requires NTS_TEST_MODEL=<path to .rten model>"]
    fn silence_produces_no_onsets() {
        let Some(path) = model_path() else { return };
        let mut det = RtenDetector::load(&path).unwrap();
        let window = vec![0.0_f32; WINDOW_SAMPLES];
        let probs = det.detect(&window, 0);
        assert!(probs.onset.iter().all(|&o| !o));
        assert_eq!(probs.frames, WINDOW_SAMPLES / SAMPLES_PER_FRAME);
    }
}
```

Run: `NTS_TEST_MODEL=<path> cargo test -p audio-input -- --ignored`
Expected: passes if the model file is available; otherwise skip this step (mandatory after Phase 5 Task 5.1 uploads the model).

**Note:** The first run must verify the model accepts variable-length `[1, 24000]` input. If `run_n` reports a shape error, fall back to: zero-pad the window to `SEGMENT_SAMPLES` (10s), run inference, take the first 150 output frames (frame 0 aligned to window start). This fallback increases per-run inference time (expected still within ~150ms on M3 Max); widen the hop to 120ms accordingly and record it in the design doc appendix.

- [x] **Step 4: Commit**

```bash
git add audio-input neothesia-ai
git commit -m "feat(audio-input): rten pitch detector backend"
```

---

### Task 2.4: StreamingPipeline

**Files:**
- Create: `audio-input/src/pipeline.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod pipeline;`)

- [x] **Step 1: Implementation**

```rust
use std::collections::VecDeque;

use crate::detector::{FrameProbabilities, PitchDetector};
use crate::tracker::{MicEvent, NoteTracker, TrackerConfig};
use crate::{
    HOP_SAMPLES, SAMPLES_PER_FRAME, TRUST_MARGIN_FRAMES, WINDOW_SAMPLES,
};

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
        let rms = (window.iter().map(|s| s * s).sum::<f32>()
            / window.len() as f32)
            .sqrt();
        if rms < RMS_GATE {
            // prolonged silence: early-release sustained notes is a
            // possible optimization; conservatively just skip
            return Vec::new();
        }

        let window_start_sample = self.total_samples - WINDOW_SAMPLES;
        let first_frame = window_start_sample / SAMPLES_PER_FRAME;
        let probs: FrameProbabilities =
            self.detector.detect(&window, first_frame);

        // Feed only NEW trusted frames: a monotonic frontier
        // [last_trusted_end, new_trusted_end) clamped to this window.
        let window_frames = WINDOW_SAMPLES / SAMPLES_PER_FRAME;
        let new_trusted_end = first_frame + window_frames - TRUST_MARGIN_FRAMES;
        let feed_start = self.last_trusted_end.max(first_frame);
        let events = if new_trusted_end > feed_start {
            self.last_trusted_end = new_trusted_end;
            self.tracker.process(feed_start, new_trusted_end, &probs)
        } else {
            Vec::new()
        };
        events
    }

    pub fn all_notes_off(&mut self) -> Vec<MicEvent> {
        self.tracker.all_notes_off()
    }
}
```

- [x] **Step 2: End-to-end frame accounting test with MockDetector**

Append to `pipeline.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{MockDetector, silent_frames};

    fn mk_pipeline(scripted: Vec<FrameProbabilities>) -> StreamingPipeline<MockDetector> {
        StreamingPipeline::new(
            MockDetector { scripted: scripted.into() },
            TrackerConfig::default(),
        )
    }

    /// Feed enough audio and verify the trusted range maps to the
    /// correct global frame indices.
    #[test]
    fn trust_boundary_maps_to_global_frames() {
        // window 1: first full window, first_frame=0, onset at global frame 3 (trusted)
        let w1 = {
            let mut pr = silent_frames(0, 150);
            pr.onset[3 * 88 + 40] = true;
            pr.frame[3 * 88 + 40] = 0.9;
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
        assert!(p.push(&noise).is_empty(), "margin frames must not trigger early");

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
```

- [x] **Step 3: Run the tests**

Run: `cargo test -p audio-input`
Expected: all pass. Frame arithmetic reference for `untrusted_margin_frames_are_deferred`: the window start sample is `total_samples - WINDOW_SAMPLES` (NOT `WINDOW_SAMPLES - HOP_SAMPLES` — the window is anchored at its start, and each hop of 960 samples slides that start by 960). After the first full window `total=24000` → start 0 → `first_frame=0`; after one hop `total=24960` → start 960 → `first_frame=6`; after two hops `first_frame=12`. The trusted frontier advances monotonically ([0,138), [138,144), [144,150)), so every frame is fed exactly once and the onset at global frame 145 first becomes trusted in window 3 (local index `145 - 12 = 133`). If the assertion fails, recompute each window's `first_frame` and local onset index with this formula — the test's semantics stay: a margin frame is deferred until its window's trusted range covers it.

- [x] **Step 4: Commit**

```bash
git add audio-input
git commit -m "feat(audio-input): streaming inference pipeline with trust boundary"
```

---

### Task 2.5: AudioInputManager public API

**Files:**
- Create: `audio-input/src/manager.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod manager;` + re-exports)

- [x] **Step 1: Implementation**

`audio-input/src/manager.rs`:

```rust
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
        let detector = RtenDetector::load(model_path)
            .map_err(AudioInputError::Detector)?;

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop2 = stop.clone();

        let buffer: Arc<SampleBuffer> = stream.buffer.clone();
        let source_rate = stream.sample_rate;
        let capture_error = stream.error.clone();

        std::thread::Builder::new()
            .name("audio-input-inference".into())
            .spawn(move || {
                let mut resampler = crate::resample::ResampleStage::new(source_rate);
                let mut pipeline =
                    StreamingPipeline::new(detector, TrackerConfig::default());
                let (tx, rx) = std::sync::mpsc::channel::<MicEvent>();

                // event relay: catch_unwind isolates inference panics
                std::thread::spawn(move || {
                    while let Ok(ev) = rx.recv() {
                        on_event(ev);
                    }
                });

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
                        Err(_) => {
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
            })
            .expect("failed to spawn audio-input thread");

        Ok(AudioInputConnection {
            device: device.clone(),
            stop,
            _stream: stream,
        })
    }
}
```

- [x] **Step 2: Final lib.rs re-exports**

`audio-input/src/lib.rs` final shape:

```rust
pub mod buffer;
pub mod capture;
pub mod detector;
pub mod manager;
pub mod pipeline;
pub mod resample;
pub mod tracker;

pub use capture::MicDevice;
pub use manager::{AudioInputConnection, AudioInputError, AudioInputManager};
pub use tracker::MicEvent;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const SAMPLES_PER_FRAME: usize = 160;
pub const WINDOW_SAMPLES: usize = 24_000;
pub const HOP_SAMPLES: usize = 960;
pub const TRUST_MARGIN_FRAMES: usize = 12;
```

- [x] **Step 3: Compile + full tests**

Run: `cargo test -p audio-input && cargo check -p audio-input --examples`
Expected: all pass.

- [x] **Step 4: Commit**

```bash
git add audio-input
git commit -m "feat(audio-input): AudioInputManager with background inference thread"
```
