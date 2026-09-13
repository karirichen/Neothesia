//! Note tracker state machine (design §5): diffs frame probabilities
//! against per-pitch state and emits NoteOn/NoteOff events equivalent
//! to MIDI input.

use crate::detector::{FIRST_MIDI_KEY, FrameProbabilities, KEY_COUNT};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicEvent {
    NoteOn {
        key: u8,
    },
    NoteOff {
        key: u8,
    },
    /// Background-thread failure (inference panic, device loss).
    Error(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct TrackerConfig {
    /// Same-pitch retrigger suppression window.
    pub onset_cooldown_frames: usize, // 25 = 250ms
    /// Frames below threshold before NoteOff.
    pub release_frames: usize, // 20 = 200ms
    /// Hard note lifetime.
    pub max_note_frames: usize, // 400 = 4s
    pub frame_release_threshold: f32, // 0.1
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
    last_onset_seen: Option<usize>,
    below_since: Option<usize>,
}

pub struct NoteTracker {
    cfg: TrackerConfig,
    states: [PitchState; KEY_COUNT],
}

impl NoteTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            states: std::array::from_fn(|_| PitchState::default()),
        }
    }

    /// Feed the newly-trusted frame range [start, end) of `probs`
    /// (which must contain those frames). Returns emitted events.
    pub fn process(
        &mut self,
        start: usize,
        end: usize,
        probs: &FrameProbabilities,
    ) -> Vec<MicEvent> {
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
                    // `last_onset_seen == None` on a fresh tracker must not
                    // suppress the pitch's first onsets of the stream.
                    let past_cooldown = st.last_onset_seen.is_none_or(|last| {
                        global_f.saturating_sub(last) >= self.cfg.onset_cooldown_frames
                    });
                    if !st.sounding && past_cooldown {
                        st.sounding = true;
                        st.onset_frame = global_f;
                        st.below_since = None;
                        events.push(MicEvent::NoteOn { key: key_of(p) });
                    } else if st.sounding
                        && global_f.saturating_sub(st.onset_frame) >= self.cfg.onset_cooldown_frames
                    {
                        // consecutive strike on same pitch: close + reopen
                        events.push(MicEvent::NoteOff { key: key_of(p) });
                        events.push(MicEvent::NoteOn { key: key_of(p) });
                        st.onset_frame = global_f;
                        st.below_since = None;
                    }
                    st.last_onset_seen = Some(global_f);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::silent_frames;

    fn window_with(
        first_frame: usize,
        frames: usize,
        specs: &[(usize, usize, f32, bool)],
    ) -> FrameProbabilities {
        // specs: (global frame, pitch, frame_prob, onset)
        let mut pr = silent_frames(first_frame, frames);
        for &(f, p, val, onset) in specs {
            let lf = f - first_frame;
            assert!(
                lf < frames,
                "spec frame {f} outside window {first_frame}..{}",
                first_frame + frames
            );
            pr.frame[lf * KEY_COUNT + p] = val;
            pr.onset[lf * KEY_COUNT + p] = onset;
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
        // window 1 only holds activation at frame 0, so below_since
        // carries over as 1; release fires at 1 + 20 = 21
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
        let cfg = TrackerConfig {
            max_note_frames: 100,
            ..Default::default()
        };
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
