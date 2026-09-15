//! Tracks which pitches the game itself is currently sounding through
//! its output (synth or MIDI out). Mic-detected onsets matching these
//! pitches are suppressed (design §6, MIDI-side echo suppression).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use midi_file::midly::MidiMessage;

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

impl SoundingNotesTracker {
    pub fn note_on(&mut self, key: u8) {
        self.inner.insert(key, State::On);
    }

    pub fn note_off(&mut self, key: u8) {
        self.inner.insert(key, State::Off { at: Instant::now() });
    }

    /// Mark every entry as released right now (stop_all semantics):
    /// keeps the ~500ms tail while un-stranding anything stuck `On`.
    pub fn all_off(&mut self) {
        let now = Instant::now();
        for state in self.inner.values_mut() {
            *state = State::Off { at: now };
        }
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

    fn midi_event(&mut self, message: &MidiMessage) {
        match message {
            MidiMessage::NoteOn { key, .. } => self.note_on(key.as_int()),
            MidiMessage::NoteOff { key, .. } => self.note_off(key.as_int()),
            _ => {}
        }
    }
}

/// The shared handle stored in `Context` and threaded through
/// `MidiPlayer`/scenes. Wraps the tracker in a Mutex and offers
/// lock-and-forward helpers so call sites stay one-liners and cannot
/// drift apart. Locking is poisoning-tolerant: a poisoned lock simply
/// stops updating tracking (suppression degrades; nothing crashes).
#[derive(Debug)]
pub struct SharedSoundingTracker(Mutex<SoundingNotesTracker>);

impl SharedSoundingTracker {
    fn lock(&self) -> std::sync::MutexGuard<'_, SoundingNotesTracker> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Update tracking for a note event being forwarded to the output.
    pub fn track_midi_event(&self, message: &MidiMessage) {
        self.lock().midi_event(message);
    }

    /// Mark everything released (call next to `stop_all`).
    pub fn all_off(&self) {
        self.lock().all_off();
    }

    /// True if `key` is currently sounding (on, or within the release
    /// tail).
    pub fn contains(&self, key: u8) -> bool {
        self.lock().contains(key)
    }
}

/// Convenience constructor for the shared tracker.
pub fn shared() -> Arc<SharedSoundingTracker> {
    Arc::new(SharedSoundingTracker(Mutex::new(
        SoundingNotesTracker::default(),
    )))
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

    #[test]
    fn all_off_unstrands_sounding_entries() {
        let mut t = SoundingNotesTracker::default();
        t.note_on(60);
        t.note_on(64);
        t.all_off();
        // within the tail both still count as sounding…
        assert!(t.contains(60) && t.contains(64));
        // …and both expire after it (simulated clock)
        let expired = Instant::now() - RELEASE_TAIL - Duration::from_millis(1);
        t.inner.insert(60, State::Off { at: expired });
        t.inner.insert(64, State::Off { at: expired });
        assert!(!t.contains(60) && !t.contains(64));
    }

    #[test]
    fn track_midi_event_maps_notes_and_ignores_others() {
        let t = shared();
        t.track_midi_event(&MidiMessage::NoteOn {
            key: 60.into(),
            vel: 100.into(),
        });
        assert!(t.contains(60));
        t.track_midi_event(&MidiMessage::NoteOff {
            key: 60.into(),
            vel: 0.into(),
        });
        assert!(t.contains(60), "release tail");
        // non-note events are ignored
        t.track_midi_event(&MidiMessage::ProgramChange { program: 1.into() });
        assert!(t.contains(60));
    }
}
