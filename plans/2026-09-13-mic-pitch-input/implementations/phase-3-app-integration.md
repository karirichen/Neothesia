# Phase 3: App Event Wiring + Echo Suppression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the MicEvent stream into `NeothesiaEvent::MidiInput` (tagged with `InputSource`), implement MIDI-side echo suppression and the "Mic is not forwarded to the synth" rule.

**Architecture:** An `InputSource` enum travels with every event; a `SoundingNotesTracker` (held by Context) is updated at every synth sounding site, and main.rs suppresses Mic NoteOns against it; Playing/FreePlay scenes skip output forwarding for the Mic source.

**Tech Stack:** Pure app-layer changes, midly MidiMessage, existing winit event loop.

**Design doc:** `plans/2026-09-13-mic-pitch-input/design.md` §4, §6

**Change map (all existing code points touched by this phase):**
- `neothesia/src/main.rs:37` `NeothesiaEvent::MidiInput` definition + `:159` consumption
- `neothesia/src/input_manager/mod.rs:45,52` (MIDI → source: Midi)
- `neothesia/src/scene/mod.rs:96,130,184` (PC keyboard/mouse → source: Keyboard/Mouse)
- `neothesia/src/scene/mod.rs:21` Scene trait `midi_event` signature + three implementations (`menu_scene/mod.rs:423`, `playing_scene/mod.rs:336`, `freeplay/mod.rs:259`)
- `neothesia/src/scene/playing_scene/midi_player.rs:93` (file events → output; tracker bookkeeping point), `:216` (user events → output; skip for Mic + bookkeep for non-Mic)
- `neothesia/src/scene/freeplay/mod.rs:262` (user events → output; skip for Mic + bookkeep for non-Mic)

---

### Task 3.1: InputSource enum and event tagging

**Files:**
- Modify: `neothesia/src/main.rs`
- Modify: `neothesia/src/input_manager/mod.rs`
- Modify: `neothesia/src/scene/mod.rs`
- Modify: `neothesia/src/scene/menu_scene/mod.rs`, `neothesia/src/scene/playing_scene/mod.rs`, `neothesia/src/scene/freeplay/mod.rs`

- [ ] **Step 1: Define the enum and extend the event in main.rs**

Above the `NeothesiaEvent` definition:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputSource {
    #[default]
    Midi,
    Keyboard,
    Mouse,
    Mic,
}
```

The `NeothesiaEvent::MidiInput` variant gains a field:

```rust
    MidiInput {
        source: InputSource,
        /// The MIDI channel that this message is associated with.
        channel: u8,
        /// The MIDI message type and associated data.
        message: MidiMessage,
    },
```

The consumption site (main.rs:159) becomes:

```rust
            NeothesiaEvent::MidiInput { source, channel, message } => {
                self.game_scene
                    .midi_event(&mut self.context, source, channel, &message);
            }
```

- [ ] **Step 2: Scene trait signature and the three implementations**

`neothesia/src/scene/mod.rs:21` trait default implementation:

```rust
    fn midi_event(
        &mut self,
        _ctx: &mut Context,
        _source: InputSource,
        _channel: u8,
        _message: &MidiMessage,
    ) {
    }
```

(Add `crate::InputSource` to the use list at the top of `scene/mod.rs`.)

All three implementations add a `source: InputSource` parameter:
- `menu_scene/mod.rs:423` → `fn midi_event(&mut self, ctx: &mut Context, _source: InputSource, channel: u8, message: &MidiMessage)`
- `playing_scene/mod.rs:336` → `fn midi_event(&mut self, _ctx: &mut Context, source: InputSource, channel: u8, message: &MidiMessage)` (body unchanged for now; Task 3.3 uses `source`)
- `freeplay/mod.rs:259` → same pattern

- [ ] **Step 3: Tag the three send sites**

Both `tx.send_event(NeothesiaEvent::MidiInput {` sites in `input_manager/mod.rs` get `source: crate::InputSource::Midi,`.

The three sites in `scene/mod.rs` (:96 PC keyboard, :130/:184 mouse) get the corresponding `source: crate::InputSource::Keyboard,` / `crate::InputSource::Mouse,`.

- [ ] **Step 4: Compile check**

Run: `cargo check -p neothesia`
Expected: passes. Any missed construction site is flagged by the compiler (verify with `rg "NeothesiaEvent::MidiInput {"` — all must carry `source`).

- [ ] **Step 5: Regression**

Run: `cargo check --workspace`
Expected: workspace compiles.

- [ ] **Step 6: Commit**

```bash
git add neothesia
git commit -m "feat(app): tag MidiInput events with InputSource"
```

---

### Task 3.2: SoundingNotesTracker

**Files:**
- Create: `neothesia/src/sounding_tracker.rs`
- Modify: `neothesia/src/main.rs` (module declaration; Context integration lands in Task 3.3)

- [ ] **Step 1: Implementation + tests**

`neothesia/src/sounding_tracker.rs`:

```rust
//! Tracks which pitches the game itself is currently sounding through
//! its output (synth or MIDI out). Mic-detected onsets matching these
//! pitches are suppressed (design §6, MIDI-side echo suppression).

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
        t.inner.insert(60, State::Off { at: Instant::now() - RELEASE_TAIL - Duration::from_millis(1) });
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
```

- [ ] **Step 2: Register the module in main.rs**

Add to the module declaration area of `neothesia/src/main.rs`:

```rust
mod sounding_tracker;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p neothesia sounding_tracker`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add neothesia
git commit -m "feat(app): sounding notes tracker for echo suppression"
```

---

### Task 3.3: Sounding-site bookkeeping (MidiPlayer + FreePlay)

**Files:**
- Modify: `neothesia/src/context.rs`
- Modify: `neothesia/src/scene/playing_scene/midi_player.rs`
- Modify: `neothesia/src/scene/playing_scene/mod.rs:97` (MidiPlayer::new call site)

- [ ] **Step 1: Context holds the shared tracker**

Add a field to `Context` in `neothesia/src/context.rs` (after `pub output_manager`):

```rust
    /// Pitches the game itself is sounding; used to suppress
    /// mic-detected ghost notes (echo suppression).
    pub sounding: std::sync::Arc<sounding_tracker::SharedSoundingTracker>,
```

Initialize in the constructor:

```rust
    sounding: crate::sounding_tracker::shared(),
```

- [ ] **Step 2: MidiPlayer bookkeeping**

Add a field to the `MidiPlayer` struct in `midi_player.rs`:

```rust
    sounding: std::sync::Arc<crate::sounding_tracker::SharedSoundingTracker>,
```

Both `new` (:22) and `new_with_lead_in` (:37) gain the parameter
`sounding: std::sync::Arc<crate::sounding_tracker::SharedSoundingTracker>` and assign it in the struct literal.

At the file-events → output site (around :93, inside the loop containing `self.output.midi_event(u4::new(channel), event.message);`), insert bookkeeping before the send:

```rust
                        if let midi_file::midly::MidiMessage::NoteOn { key, .. } =
                            event.message
                        {
                            if let Ok(t) = self.sounding.lock() {
                                t.note_on(key.as_int());
                            }
                        }
                        if let midi_file::midly::MidiMessage::NoteOff { key, .. } =
                            event.message
                        {
                            if let Ok(t) = self.sounding.lock() {
                                t.note_off(key.as_int());
                            }
                        }
                        self.output.midi_event(u4::new(channel), event.message);
```

(Keep the surrounding statements unchanged; align field names with the actual code if they differ.)

The user-event forward site (`user_midi_event`, around :215) becomes:

```rust
    pub fn user_midi_event(
        &mut self,
        channel: u8,
        message: &MidiMessage,
        source: crate::InputSource,
    ) {
        // Mic source is not forwarded: the real piano is the sound
        // source; a synth follow-along would create echo
        if source == crate::InputSource::Mic {
            self.play_along.midi_event(MidiEventSource::User, message);
            return;
        }

        match message {
            MidiMessage::NoteOn { key, .. } => {
                if let Ok(t) = self.sounding.lock() {
                    t.note_on(key.as_int());
                }
            }
            MidiMessage::NoteOff { key, .. } => {
                if let Ok(t) = self.sounding.lock() {
                    t.note_off(key.as_int());
                }
            }
            _ => {}
        }
        self.output.midi_event(u4::new(channel), *message);
        self.play_along.midi_event(MidiEventSource::User, message);
    }
```

`should_forward_human_event` (:226) keeps its existing semantics (pass-through of non-note events to MIDI output) — untouched.

- [ ] **Step 3: Pass the tracker at the call sites**

The `MidiPlayer::new(` call at `playing_scene/mod.rs:97` gains the argument `ctx.sounding.clone(),` (the `ctx` parameter of `PlayingScene::new` is available; do the same for `new_with_lead_in` if it has other call sites — verify with rg).

Update the `midi_event` implementation at `playing_scene/mod.rs:336`:

```rust
    fn midi_event(
        &mut self,
        _ctx: &mut Context,
        source: crate::InputSource,
        channel: u8,
        message: &MidiMessage,
    ) {
        self.player.user_midi_event(channel, message, source);
        self.keyboard.user_midi_event(message);
    }
```

- [ ] **Step 4: FreePlay bookkeeping + Mic skip**

`freeplay/mod.rs:259`:

```rust
    fn midi_event(
        &mut self,
        ctx: &mut Context,
        source: crate::InputSource,
        channel: u8,
        message: &MidiMessage,
    ) {
        self.recorder.push_event(channel, *message);
        self.keyboard.user_midi_event(message);

        if source != crate::InputSource::Mic {
            match message {
                MidiMessage::NoteOn { key, .. } => {
                    if let Ok(t) = ctx.sounding.lock() {
                        t.note_on(key.as_int());
                    }
                }
                MidiMessage::NoteOff { key, .. } => {
                    if let Ok(t) = ctx.sounding.lock() {
                        t.note_off(key.as_int());
                    }
                }
                _ => {}
            }
            ctx.output_manager
                .connection()
                .midi_event(channel.into(), *message);
        }
    }
```

(Keep whatever follows the original forward in the function body unchanged.)

- [ ] **Step 5: Compile check**

Run: `cargo check -p neothesia`
Expected: passes.

- [ ] **Step 6: Commit**

```bash
git add neothesia
git commit -m "feat(app): track sounding pitches at synth forward sites"
```

---

### Task 3.4: Mic-side suppression + AudioInputManager wiring

**Files:**
- Modify: `neothesia/src/main.rs` (suppression in user_event + Context field)
- Modify: `neothesia/src/context.rs` (audio_input connection field)

- [ ] **Step 1: Suppression logic in main.rs**

The `NeothesiaEvent::MidiInput` branch of `user_event` (signature updated in Task 3.1) gains suppression:

```rust
            NeothesiaEvent::MidiInput { source, channel, message } => {
                if source == InputSource::Mic {
                    if let midly::MidiMessage::NoteOn { key, .. } = message {
                        if self.context.sounding.lock().map(|t| t.contains(key.as_int())).unwrap_or(false) {
                            log::debug!("suppressed ghost onset {}", key.as_int());
                            return;
                        }
                    }
                }
                self.game_scene
                    .midi_event(&mut self.context, source, channel, &message);
            }
```

- [ ] **Step 2: Context integrates the audio input**

Add a field to `Context` in `context.rs`:

```rust
    pub audio_input: Option<audio_input::AudioInputConnection>,
```

Add to `neothesia/Cargo.toml` `[dependencies]`:

```toml
audio-input.workspace = true
```

(The workspace dependency was defined in Phase 1.) Initialize with `audio_input: None` in the constructor (the settings UI in Phase 4 establishes real connections; startup restore also lands in Phase 4).

- [ ] **Step 3: MicEvent → NeothesiaEvent mapping helper**

Add to `main.rs`:

```rust
fn mic_event_to_neothesia(event: audio_input::MicEvent) -> Option<NeothesiaEvent> {
    use midi_file::midly::MidiMessage;
    match event {
        audio_input::MicEvent::NoteOn { key } => Some(NeothesiaEvent::MidiInput {
            source: InputSource::Mic,
            channel: 0,
            message: MidiMessage::NoteOn { key: key.into(), vel: 100.into() },
        }),
        audio_input::MicEvent::NoteOff { key } => Some(NeothesiaEvent::MidiInput {
            source: InputSource::Mic,
            channel: 0,
            message: MidiMessage::NoteOff { key: key.into(), vel: 0.into() },
        }),
        audio_input::MicEvent::Error(msg) => {
            log::error!("audio input error: {msg}");
            None
        }
    }
}
```

(The connection code in Phase 4 reuses it.)

- [ ] **Step 4: Workspace compile + tests**

Run: `cargo check --workspace && cargo test -p neothesia sounding_tracker`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add neothesia Cargo.toml Cargo.lock
git commit -m "feat(app): mic-side echo suppression and event mapping"
```
