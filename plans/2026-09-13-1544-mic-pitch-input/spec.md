# Microphone Pitch Detection Input (Mic Pitch Input) — Specification

- Date: 2026-09-13
- Status: All sections (§1-§4) confirmed with the stakeholder
- Goal: Let acoustic pianos with no USB/MIDI interface (performance/home instruments) be playable in Neothesia

## 1. Background & Goals

Neothesia currently supports only MIDI controllers as piano input. This design adds a
**microphone audio input** path: the built-in microphone captures the piano sound, an ML
polyphonic pitch detector converts it into NoteOn/NoteOff events equivalent to MIDI input,
giving acoustic piano users the same gameplay experience (scoring, keyboard highlight,
FreePlay) as MIDI keyboard users.

### Confirmed requirements

| Dimension | Decision |
|---|---|
| Capture device | Laptop built-in microphone (environmental noise must be considered) |
| Polyphony | Chords must be supported (polyphonic detection) |
| Latency target | Key press to in-game highlight ≤200ms (typical ~220ms, tuned empirically) |
| Tech route | ML model (rten, Basic Pitch style), reusing neothesia-ai infrastructure |
| Inference backend | CPU first (M-series NEON is sufficient), replaceable-backend seam reserved |
| Speaker echo | MIDI-side suppression + Mic events not forwarded to synth |
| Model distribution | Download on first use (~8MB rten model) |
| Scene coverage | Playing + FreePlay + settings-page input-source UI |

### Latency bottleneck note

End-to-end latency is dominated by the **trust margin** (the model needs ~100-150ms of
context after an onset to confirm it), not by compute. A GPU would compress the ~40ms
inference step to ~10ms, which is drowned by the trust margin; and the ort/candle routes
bring cross-platform packaging complexity or a model rewrite. Therefore GPU is not adopted
in v1; only a replacement seam is reserved.

## 2. Existing architecture integration points (exploration findings)

- All inputs (MIDI / PC keyboard / mouse) converge into
  `NeothesiaEvent::MidiInput { channel, message }` (`neothesia/src/main.rs:37`);
  scenes consume them uniformly via `Scene::midi_event()`. Audio input plugs in
  seamlessly as long as it can produce NoteOn/NoteOff.
- `cpal` is already a workspace dependency (currently output-only, but it also supports
  input capture).
- `neothesia-ai` already contains offline audio→MIDI transcription (rten + Basic Pitch
  style model, 16kHz mono, onset/offset/frame outputs), currently a standalone CLI
  (10s segments, not real-time).
- The play-along scoring logic is naturally latency-friendly: early presses get a 500ms
  grace window (`midi_player.rs` PlayAlong::user_pressed_recently), late presses remain
  valid for the whole note duration. A ~200ms detection delay fits inside the existing
  grace windows.

## 3. Approach selection record

| Approach | Verdict |
|---|---|
| A. Sliding-window streaming inference | **Adopted**. Best balance of latency/accuracy/complexity; reuses neothesia-ai |
| B. Hybrid DSP gate + ML confirmation | Rejected. Double-system complexity is not worth it; can be a future optimization path |
| C. Chunked offline reuse | Rejected. 2-3s latency fails the requirement |

Inference backend: rten CPU (optimized for ARM NEON) first; the pipeline depends on a
`PitchDetector` trait so an ort/CoreML or candle/Metal backend can be added later without
touching the pipeline.

## 4. Overall architecture & crate layout (design §1, confirmed)

```
┌─ neothesia (app) ──────────────────────────────────────────┐
│  InputManager (MIDI, existing)                              │
│  AudioInputManager (new) ──┐                                │
│                            ├─→ NeothesiaEvent::MidiInput {  │
│  PC keyboard/mouse (existing)─┘  source, channel, message } │
│                             ↓                               │
│  Scene layer (Playing/FreePlay/Menu) — zero-change consume  │
└─────────────────────────────────────────────────────────────┘
```

| Crate | Change |
|---|---|
| `audio-input` **(new)** | Microphone capture (cpal), resampling, streaming inference, note tracker. Exposes only `AudioInputManager` (API style mirrors `midi_io::MidiInputManager`: `devices()` / `connect()` / disconnect notification) |
| `neothesia-ai` | Refactor: core inference logic (enframe/deframe/note detection) extracted into a library reused by `audio-input`; `main.rs` becomes a thin CLI shell. **Its existing offline functionality is unchanged** |
| `midi-io` | Untouched |
| `neothesia` (app) | `NeothesiaEvent::MidiInput` gains a `source: InputSource` enum (`Keyboard`/`Mouse`/`Midi`/`Mic`); new `AudioInputManager` wiring; settings page gains an input-source section |

Concurrency model: `audio-input` owns its capture callback thread + inference thread and
sends events through the existing `EventLoopProxy`; the winit event loop naturally
serializes them, so the scene layer stays lock-free.

## 5. Core pipeline (design §2, confirmed)

```
cpal callback thread                inference thread (loop)
────────────                        ─────────────────────────────────────
input stream f32 (default device    every 60ms (hop) take the last 1.5s window
config, usually 44.1/48kHz,     →   ├─ energy gate: RMS < threshold → skip this run (CPU saver)
downmixed to mono)                 ├─ resample to 16kHz (rubato, anti-aliased)
        ↓                          ├─ PitchDetector::detect(window)
   SPSC buffer (~3s capacity,      │    └─ rten model → onset/frame/offset matrices (100fps)
   drop-oldest on overflow)        └─ NoteTracker: diff newly-trusted frames vs emitted state
                                       ├─ onset rising edge > 0.3 → NoteOn
                                       ├─ pitch activation < 0.1 for 200ms → NoteOff
                                       ├─ same-pitch dedup within 250ms (re-trigger guard)
                                       └─ force NoteOff after 4s max (reverb-tail insurance)
        events → EventLoopProxy → NeothesiaEvent::MidiInput { source: Mic }
```

Key parameters (all configurable, initial values above): window 1.5s / hop 60ms /
trust margin 120ms.

Latency budget: mic buffer ~30ms + inference ~40ms (M3 Max, 1.5s window) + trust margin
120ms + half hop 30ms ≈ ~220ms typical; compressing the trust margin to 80ms reaches
~180ms at the cost of slightly lower onset accuracy — to be tuned empirically.

Replaceable backend seam:

```rust
trait PitchDetector {
    fn detect(&mut self, window: &[f32], first_frame: usize) -> FrameProbabilities;
}
```

The pipeline depends only on this trait; the rten implementation is the default
(reusing the library extracted from neothesia-ai).

Velocity: fixed at 100 in v1; the model's velocity output is a future enhancement.

## 6. Echo suppression (design §3, confirmed)

- New `SoundingNotesTracker` at the app layer: tracks the set of pitches the game itself
  is currently sounding through the output manager (a synth NoteOff removes the pitch only
  after a ~500ms delay, covering release tails).
- A Mic-sourced NoteOn that hits this set is dropped outright (no play-along update, no
  keyboard highlight).
- **Mic-sourced events are not forwarded to the synth/MIDI output** (the real piano is
  the sound source; a synth follow-along would create a new echo source). The existing
  `user_midi_event → output` forwarding in the Playing scene is skipped for the Mic source.
- MIDI keyboard users are completely unaffected (source=Midi forwards as before).
- Known blind spot: the user playing the same pitch as the accompaniment simultaneously
  gets falsely suppressed. Low probability; accepted and documented.

## 7. Settings UI & configuration

- A "Microphone" section next to the MIDI port list on the input settings page: device
  dropdown (cpal input-device enumeration) + enable toggle. No level meter (YAGNI).
- Persisted into the existing ron config: `audio_input.enabled` / `audio_input.device`.
- Enabling with no model downloaded triggers the download flow (status + retry on failure).

## 8. Model distribution

- On first enable, download the ~8MB rten model into the user data directory
  (macOS: `~/Library/Application Support/neothesia/models/`, located via the `dirs` crate).
- Download source: a GitHub Releases attachment of this fork (the converted model is
  uploaded once during implementation); URL + SHA256 are code constants, verified after
  download.
- Checksum/network failure → error message on the settings page + retry, no crash.
- Corrupted model file (load failure) → auto-delete and re-download.

## 9. Error handling & platform permissions

- macOS: `NSMicrophoneUsageDescription` in `Info.plist` + entitlement; on denial,
  UI prompt guiding the user to System Settings.
- Microphone hot-unplug / no device → notification; reconnectable from the settings page.
- Inference thread isolated with `catch_unwind`: on panic, mic input is auto-disabled
  with a notification; the main program is unaffected.
- Ring-buffer overflow (inference occasionally falling behind) → drop oldest audio
  (a detection gap is preferable to stalling).

## 10. Test strategy (design §4, confirmed)

1. **Unit tests** (`audio-input`): synthetic frame-probability sequences fed into
   `NoteTracker` to validate the state machine (trigger/dedup/timeout/force-off);
   resampler frequency preservation against a known sine; `SoundingNotesTracker`
   add/expire behavior.
2. **Offline accuracy regression (core, no microphone needed)**: `test.mid` → rendered
   to wav by the existing synth → samples fed into `PitchDetector` → compare onset hit
   rate against the source MIDI (±50ms same pitch); mix in the accompaniment track +
   noise to validate suppression; collect the onset detection latency distribution and
   assert p95 < 250ms. Cases not requiring the model file run in CI; model-dependent
   cases are `#[ignore]` and run locally.
3. **Manual test checklist on real hardware**: scales, chords, pedaled pieces, soft
   dynamics, laptop placement variations (macOS built-in microphone first).

## 11. Known limitations (documented, not bugs)

- Sustain pedal state is unknowable (no CC64); NoteOff is energy-based and later than
  the physical key release.
- Velocity fixed at 100.
- Simultaneous same-pitch strikes with the accompaniment get falsely suppressed (§6 blind spot).
- Voice/ambient sound near the microphone may produce phantom notes (mitigated by the
  energy gate, not fully solved).
- Accuracy degrades on very dense chords (>6 notes).
- macOS tested first; Linux/Windows theoretically supported (cpal) but unverified.
- Threshold parameters are configurable only via config file; no GUI tuning UI.

## 12. Milestone draft (for writing-plans to detail)

1. Phase 1: `neothesia-ai` library extraction + `audio-input` crate scaffold
   (capture/resample/SPSC buffer + unit tests)
2. Phase 2: streaming inference + NoteTracker + `PitchDetector` trait
   (synthetic-audio end-to-end tests)
3. Phase 3: event wiring into the app (`InputSource` tag, echo suppression,
   no synth forwarding)
4. Phase 4: settings UI + config persistence + model downloader
5. Phase 5: offline accuracy regression + latency tuning + platform permissions/packaging

## Appendix A: Implementation deviations

| Design as written | Implemented as | Reason |
|---|---|---|
| Lock-free ring buffer | Mutex<VecDeque> SPSC semantics | Write side locks once per ~20ms, read side once per 60ms — negligible contention; avoids ringbuf API version risk |
| hop 64ms | hop 60ms (960 samples = 6 frames) | Sits on the 10ms frame grid, avoiding fractional-frame accounting |
| Download progress bar | Status text "Downloading..." | on_async is a single completion callback; streaming progress needs extra plumbing — YAGNI |
| Streaming onset detection | Simple threshold (>0.3) initially | The offline path's monotonic-neighbour refinement is deferred to the tuning stage (see Phase 5 Task 5.2 parameter log) |
| Accuracy regression via test.mid rendered through the synth | Synthetic piano-ish harmonic sines (5.2) | Regression works without a soundfont rendering pipeline; the synth-rendered version is a future enhancement |
| Hot-unplug toast notification | log + MicEvent::Error event (v1 logs only) | No global toast infrastructure in v1; the Error event already enters the event stream; UI treatment is future work |
| `devices()` filters via `name().ok()` | warn + empty Vec on enumeration error (connect still propagates) | cpal 0.18 removed `name()`; enumeration failure is non-fatal for the picker UI |
| Pipeline feeds per-window trusted range `first_frame..first_frame+trusted_end` | Monotonic trusted frontier `[last_trusted_end, new_trusted_end)` + tracker monotonicity assert | Review found overlap re-feed underflows `global_f - onset_frame` (usize wrap → silent note loss in release builds) |
| Clean stop exits the inference loop silently | Clean stop emits `all_notes_off` before thread exit | Review found settings-toggle-off mid-note would leave keys highlighted forever |
| Crate exposes only `AudioInputManager` | All modules are `pub mod` + lib re-exports | Phase 4 needs `model_store` and constants; internal coherence is unaffected |
| SPSC capacity ~3s | 4s (capture.rs) | Margin for 20ms-poll jitter; drop-oldest backstop unchanged |
| Mic events recorded in FreePlay are previewed through the synth | Intentional | A recording captures what the player played; preview playback synthesizing it is the same semantics as MIDI-keyboard recordings |

## Appendix B: Real-hardware manual test checklist (macOS, MacBook built-in microphone)

- [ ] Single-note C-major scale up/down; every note lights up/goes dark correctly
- [ ] Triads / seventh chords struck together; all keys light up
- [ ] Legato same-note repetition (same pitch >250ms apart) re-triggers
- [ ] Sustained note with pedal: NoteOff is later than the key release but force-cut within 4s
- [ ] Both pp and ff dynamics trigger
- [ ] Laptop placement: screen facing the music stand vs. side placement, both usable
- [ ] Speaker accompaniment vs. headphones: with speakers, the accompaniment produces no ghost highlights
- [ ] Guidance prompt appears after microphone permission denial (bare-terminal run: grant the parent terminal in System Settings)
- [ ] After unplugging/disabling the microphone, the error is logged and the connection thread exits; re-enabling the toggle in settings recovers (toast UI in a later version)
| Config keys `audio_input.enabled/device` | `mic.enabled/device` | Section name matches the sibling config sections' domain naming |
| Latency regression asserts p95 < 250ms | p95 < 400ms | First-window construction adds up to 1s apparent latency for early onsets; steady-state (probe at 2.0s) is what the budget governs; real budget validation is Phase 5 on-hardware tuning |
| Info.plist + entitlement | Info.plist only | CI builds are unsigned; a notarized/hardened-runtime build would additionally need the audio-input entitlement — tracked for whenever signing lands |
| Suppression validated against accompaniment+noise mix (§10) | Manual checklist item only | The automated counterpart needs a synth-rendered mix harness — future enhancement (same family as the sines-vs-synth-render deviation) |
| Model ~8MB, hop 60ms, inference ~40ms | Kong 2020 model is 147MB fp32; hop 200ms; inference ~150ms (rten 0.26) | The 8MB estimate assumed a lighter model; Kong's CRNN is 36M params. rten upgraded 0.24→0.26 (4.3x inference speedup) and hop widened to keep CPU ≤0.75x — measured steady-state onset latency p95 = 200ms in the offline harness |
| Model input fixed [1, 160000] | Input dim patched to dynamic in the ONNX graph before conversion (validated numerically against the fixed graph) | Streaming windows are [1, 24000]; the zero-padding fallback (6.7x compute) would not fit realtime |
| First-use download from GitHub Releases | Same, but CDN slow under mainland-China networks | Anonymous URL verified via API; users behind the GFW may need a proxy for the 147MB first-use download |
