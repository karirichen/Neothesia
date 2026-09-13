# Phase 5: Accuracy Regression + Latency Tuning + Platform Packaging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offline accuracy/latency regression tests (no microphone needed), model release and constant backfill, macOS microphone permission, and the real-hardware manual test checklist.

**Architecture:** Detection quality is locked in by a "synthetic audio → pipeline → compare against known notes" offline harness (`#[ignore]`-gated, needs a local model); packaging only touches the CI manifest and bundle metadata.

**Tech Stack:** Sine synthesis, the existing rten model, GitHub Releases, winit/CI bundle.

**Design doc:** `plans/2026-09-13-mic-pitch-input/design.md` §10, §11

---

### Task 5.1: Model upload and constant backfill (operator task)

**Files:**
- Modify: `audio-input/src/model_store.rs`

- [ ] **Step 1: Locate/convert the model file**

The rten model used by neothesia-ai is converted from the Basic Pitch ONNX model. Use an existing local `.rten` file if available; otherwise convert the upstream ONNX model with `rten-convert` (see the model-conversion section of https://github.com/robertknight/rten). Under the GFW, read `~/docs/network-access-china.md` first and configure the proxy before fetching from GitHub.

- [ ] **Step 2: Compute the SHA256**

Run: `shasum -a 256 <path-to>/piano_transcription.rten`
Expected: a 64-hex-character digest (write it down).

- [ ] **Step 3: Upload to a GitHub Release**

Create a Release tagged `models` (or `models-v1`) on this fork and upload `piano_transcription.rten` as an attachment. Confirm anonymous download works:

Run: `curl -L -o /tmp/model-check.rten "<MODEL_URL>" && shasum -a 256 /tmp/model-check.rten`
Expected: matches the Step 2 digest.

- [ ] **Step 4: Backfill the constants**

Replace `MODEL_URL` in `model_store.rs` (fixing the repo name/tag if the placeholder differs) and `MODEL_SHA256` with the real values. Remove the note about the REPLACE placeholder.

- [ ] **Step 5: Commit**

```bash
git add audio-input
git commit -m "chore(audio-input): pin model release URL and checksum"
```

---

### Task 5.2: Sine-chord detection regression test

**Files:**
- Create: `audio-input/tests/integration.rs`

- [ ] **Step 1: Write the tests**

`audio-input/tests/integration.rs`:

```rust
//! Offline detection regression. Requires the real model:
//! NTS_TEST_MODEL=<path> cargo test -p audio-input --test integration -- --ignored --nocapture

use audio_input::detector::rten_backend::RtenDetector;
use audio_input::tracker::{MicEvent, TrackerConfig};
use audio_input::{HOP_SAMPLES, SAMPLES_PER_FRAME, TARGET_SAMPLE_RATE, WINDOW_SAMPLES};

const SAMPLE_RATE: u32 = TARGET_SAMPLE_RATE;

fn midi_hz(key: u8) -> f32 {
    440.0 * 2.0_f32.powf((key as f32 - 69.0) / 12.0)
}

/// ADSR-ish tone: fast attack, exponential decay, plus a 0.4x second
/// harmonic to emulate piano overtones.
fn pianoish_tone(freq: f32, start: usize, seconds: f32, gain: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE as f32) as usize;
    let mut out = vec![0.0_f32; start];
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let env = (-t * 1.5).exp() * (1.0 - (-t * 50.0).exp());
        let s = (2.0 * std::f32::consts::PI * freq * t).sin()
            + 0.4 * (2.0 * std::f32::consts::PI * 2.0 * freq * t).sin();
        out.push(s * env * gain);
    }
    out
}

fn run_pipeline(samples: &[f32]) -> (Vec<MicEvent>, Vec<usize>) {
    let model_path = std::env::var("NTS_TEST_MODEL").expect("NTS_TEST_MODEL not set");
    let det = RtenDetector::load(model_path.as_ref()).unwrap();
    let mut p = StreamingPipeline::new(det, TrackerConfig::default());

    let mut events = Vec::new();
    let mut onsets_at = Vec::new(); // sample index at detection time
    let mut fed = 0usize;
    for chunk in samples.chunks(HOP_SAMPLES) {
        // events returned by push() are produced once the chunk is fed —
        // recording the sample position gives a latency upper bound
        fed += chunk.len();
        for ev in p.push(chunk) {
            if matches!(ev, MicEvent::NoteOn { .. }) {
                onsets_at.push(fed);
            }
            events.push(ev);
        }
    }
    let _ = (SAMPLES_PER_FRAME, WINDOW_SAMPLES);
    (events, onsets_at)
}

#[test]
#[ignore = "requires NTS_TEST_MODEL"]
fn detects_three_note_chord() {
    // A4 + C#5 + E5 (A major), all struck at 0.5s
    let start = (0.5 * SAMPLE_RATE as f32) as usize;
    let keys = [69u8, 73, 76];

    let mut samples = vec![0.0_f32; start + (3.0 * SAMPLE_RATE as f32) as usize];
    for &k in &keys {
        let tone = pianoish_tone(midi_hz(k), start, 2.5, 0.3);
        for (i, s) in tone.into_iter().enumerate() {
            samples[i] += s;
        }
    }

    let (events, _) = run_pipeline(&samples);

    let detected: std::collections::BTreeSet<u8> = events
        .iter()
        .filter_map(|e| match e {
            MicEvent::NoteOn { key } => Some(*key),
            _ => None,
        })
        .collect();

    for &k in &keys {
        assert!(
            detected.contains(&k),
            "key {k} not detected, got: {detected:?}"
        );
    }
}

#[test]
#[ignore = "requires NTS_TEST_MODEL"]
fn onset_latency_p95_under_400ms() {
    let start = (0.5 * SAMPLE_RATE as f32) as usize;
    let tone = pianoish_tone(midi_hz(60), start, 2.0, 0.3);
    let total = start + (2.5 * SAMPLE_RATE as f32) as usize;
    let mut samples = vec![0.0_f32; total];
    for (i, s) in tone.into_iter().enumerate() {
        samples[i] += s;
    }

    let (_, onsets_at) = run_pipeline(&samples);
    assert!(!onsets_at.is_empty(), "no onset detected");

    let latencies_ms: Vec<f32> = onsets_at
        .iter()
        .map(|&fed| (fed as f32 - start as f32) / SAMPLE_RATE as f32 * 1000.0)
        .collect();
    let p95 = percentile(&latencies_ms, 0.95);
    eprintln!("onset latencies (ms): {latencies_ms:?}, p95={p95}");
    assert!(p95 < 400.0, "p95 latency {p95}ms exceeds budget");
}

fn percentile(sorted_input: &[f32], p: f32) -> f32 {
    let mut v = sorted_input.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f32 - 1.0) * p).round() as usize;
    v[idx.min(v.len() - 1)]
}
```

Note: `run_pipeline` uses `StreamingPipeline` directly — add `use audio_input::StreamingPipeline;` to the imports at the top of the file (it is re-exported at crate root; adjust to `use audio_input::pipeline::StreamingPipeline;` if not).

- [ ] **Step 2: Run**

Run: `NTS_TEST_MODEL=<path> cargo test -p audio-input --test integration -- --ignored --nocapture`
Expected: both tests pass. If `detects_three_note_chord` misses keys: first lower `RMS_GATE` (a 0.3 gain may be too quiet), then lower `ONSET_THRESHOLD` (0.3 → 0.25) — change one parameter at a time and record each step. If the model rejects variable-length input with a shape error, enable the zero-padding fallback documented in Phase 2 Task 2.3.

- [ ] **Step 3: Commit**

```bash
git add audio-input
git commit -m "test(audio-input): offline chord detection and latency regression"
```

---

### Task 5.3: macOS microphone permission and packaging metadata

**Files:**
- Modify: the macOS packaging flow under `.github/workflows/` (identify the exact file on site)

- [ ] **Step 1: Find the bundle manifest**

Run: `fd -t f . .github/workflows && rg -n "Info.plist|NSMicrophone|cargo-bundle|app$" .github/workflows`
Expected: locate the macOS .app build step (either hand-assembled or cargo-bundle).

- [ ] **Step 2: Inject the permission declaration**

Wherever Info.plist is generated, add:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Neothesia uses the microphone to detect the notes you play on an acoustic piano.</string>
```

If CI has no Info.plist generation (bare binary releases), create a `neothesia/Info.plist` template for macOS and copy it in during the .app assembly step; also document that bare `cargo run` has no bundle, so macOS mic permission depends on the parent process — testing requires a `cargo bundle` artifact or manually allowing the terminal in System Settings.

- [ ] **Step 3: Verify**

Locally (if a macOS packaging script exists) or at minimum validate the CI YAML:
Run: `cargo check --workspace`
After pushing, watch the macOS Actions job go green.

- [ ] **Step 4: Commit**

```bash
git add .github
git commit -m "ci(macos): declare microphone usage in app bundle"
```

---

### Task 5.4: Design doc appendix update + real-hardware manual checklist

**Files:**
- Modify: `plans/2026-09-13-mic-pitch-input/design.md` (append the appendix)

- [ ] **Step 1: Append the implementation-deviation appendix**

Append to design.md:

```markdown
## Appendix A: Implementation deviations

| Design as written | Implemented as | Reason |
|---|---|---|
| Lock-free ring buffer | Mutex<VecDeque> SPSC semantics | Write side locks once per ~20ms, read side once per 60ms — negligible contention; avoids ringbuf API version risk |
| hop 64ms | hop 60ms (960 samples = 6 frames) | Sits on the 10ms frame grid, avoiding fractional-frame accounting |
| Download progress bar | Status text "Downloading..." | on_async is a single completion callback; streaming progress needs extra plumbing — YAGNI |
| Streaming onset detection | Simple threshold (>0.3) initially | The offline path's monotonic-neighbour refinement is deferred to the tuning stage (see Phase 5 Task 5.2 parameter log) |
| Accuracy regression via test.mid rendered through the synth | Synthetic piano-ish harmonic sines (5.2) | Regression works without a soundfont rendering pipeline; the synth-rendered version is a future enhancement |
| Hot-unplug toast notification | log + MicEvent::Error event (v1 logs only) | No global toast infrastructure in v1; the Error event already enters the event stream; UI treatment is future work |

## Appendix B: Real-hardware manual test checklist (macOS, MacBook built-in microphone)

- [ ] Single-note C-major scale up/down; every note lights up/goes dark correctly
- [ ] Triads / seventh chords struck together; all keys light up
- [ ] Legato same-note repetition (same pitch >250ms apart) re-triggers
- [ ] Sustained note with pedal: NoteOff is later than the key release but force-cut within 4s
- [ ] Both pp and ff dynamics trigger
- [ ] Laptop placement: screen facing the music stand vs. side placement, both usable
- [ ] Speaker accompaniment vs. headphones: with speakers, the accompaniment produces no ghost highlights
- [ ] Guidance prompt appears after microphone permission denial
- [ ] After unplugging/disabling the microphone, the error is logged and the connection thread exits; re-enabling the toggle in settings recovers (toast UI in a later version)
```

- [ ] **Step 2: Execute the manual checklist**

Verify each item on real hardware; for failures, return to the corresponding phase and tune (threshold parameters live in `TrackerConfig`, `RMS_GATE`, `TRUST_MARGIN_FRAMES`).

- [ ] **Step 3: Commit**

```bash
git add plans/2026-09-13-mic-pitch-input
git commit -m "docs(mic-input): implementation deviations and manual test checklist"
```

---

## Phase 5 completion criteria

- `NTS_TEST_MODEL=... cargo test -p audio-input -- --ignored` all green
- `cargo test --workspace` all green (excluding ignored)
- The GitHub Release model downloads and its SHA256 verifies
- The macOS CI bundle carries the microphone permission declaration
- design.md Appendices A/B in place; the manual checklist executed
