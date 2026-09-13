//! Tracks which pitches the game itself is currently sounding through
//! its output (synth or MIDI out). Mic-detected onsets matching these
//! pitches are suppressed (design §6, MIDI-side echo suppression).

// Wired into Context in task 3.3; remove this attribute then.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a pitch still counts as "sounding" after its NoteOff,
/// covering synth release tails and reverb.
const RELEASE_TAIL: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Copy)]
enum State {
    On,
    Off { at: Instant },
}

#[derive(Debug, Default)]
pub struct SoundingNotesTracker {
    inner: HashMap<u8, State>,
}

/// Arc<Mutex<..>> alias — shared between Context, MidiPlayer and scenes.
pub type SharedSoundingTracker = Mutex<SoundingNotesTracker>;

impl SoundingNotesTracker {
    pub fn note_on(&mut self, key: u8) {
        self.inner.insert(key, State::On);
    }

    pub fn note_off(&mut self, key: u8) {
        self.inner.insert(key, State::Off { at: Instant::now() });
    }

    /// True if `key` is currently sounding (still on, or within the
    /// release tail after off).
    pub fn contains(&self, key: u8) -> bool {
        match self.inner.get(&key) {
            Some(State::On) => true,
            Some(State::Off { at }) => at.elapsed() < RELEASE_TAIL,
            None => false,
        }
    }
}

/// Convenience constructor for the shared tracker.
pub fn shared() -> std::sync::Arc<SharedSoundingTracker> {
    std::sync::Arc::new(Mutex::new(SoundingNotesTracker::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_then_off_lingers_then_clears() {
        let mut t = SoundingNotesTracker::default();
        t.note_on(60);
        assert!(t.contains(60));
        t.note_off(60);
        assert!(t.contains(60), "still sounding within the release tail");
        // we can't fast-forward the clock → simulate expiry directly
        t.inner.insert(
            60,
            State::Off {
                at: Instant::now() - RELEASE_TAIL - Duration::from_millis(1),
            },
        );
        assert!(!t.contains(60));
    }

    #[test]
    fn unknown_key_absent() {
        let t = SoundingNotesTracker::default();
        assert!(!t.contains(61));
    }

    #[test]
    fn retrigger_after_tail() {
        let mut t = SoundingNotesTracker::default();
        t.note_on(60);
        t.note_off(60);
        t.note_on(60);
        assert!(t.contains(60));
    }
}
