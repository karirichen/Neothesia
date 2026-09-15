# Phase 1: neothesia-ai Library Extraction + audio-input Crate Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Extract neothesia-ai's inference core into a reusable library and build the audio-input crate's capture/buffer/resampling infrastructure (no model inference yet).

**Architecture:** neothesia-ai splits into lib + thin bin; the new crate `audio-input` provides an SPSC audio buffer and incremental resampling (arbitrary device rate → 16kHz mono). Everything in this phase is verifiable with pure unit tests — no microphone or model file needed.

**Tech Stack:** Rust workspace, cpal (device enumeration and stream; this phase only enumerates + compiles), rubato (resampling), existing `cargo test`.

**Design doc:** `plans/2026-09-13-1544-mic-pitch-input/spec.md`

---

### Task 1.1: Split neothesia-ai into lib + bin

**Files:**
- Create: `neothesia-ai/src/lib.rs`
- Create: `neothesia-ai/src/transcription.rs`
- Modify: `neothesia-ai/src/main.rs`

- [x] **Step 1: Create lib.rs and move reusable functions into a public module**

`neothesia-ai/src/lib.rs`:

```rust
//! Neothesia AI: audio to piano transcription core.
//! Shared between the offline CLI (main.rs) and the realtime
//! audio-input pipeline of the game.

pub const FRAMES_PER_SECOND: usize = 100;
pub const SAMPLE_RATE: u32 = 16000;
/// Offline CLI segmentation size (10s). NOT used by streaming.
pub const SEGMENT_SAMPLES: usize = SAMPLE_RATE as usize * 10;

mod transcription;

pub use transcription::{
    create_midi_file, deframe, enframe, get_binarized_output_from_regression,
    is_monotonic_neighbour, note_detection_with_onset_offset_regress,
};
```

Create `neothesia-ai/src/transcription.rs` containing (moved verbatim from the current `main.rs`, only visibility made `pub`):
- `pub fn enframe(...)` (main.rs:81-101)
- `pub fn deframe(...)` (main.rs:104-142)
- `pub fn get_binarized_output_from_regression(...)` (main.rs:144-175)
- `pub fn is_monotonic_neighbour(...)` (main.rs:177-193)
- `pub fn note_detection_with_onset_offset_regress(...)` (main.rs:195-237)
- `fn note_detection_with_onset_offset_regress_inner(...)` (main.rs:239-333, stays private)
- `pub fn create_midi_file(...)` (main.rs:335-407)

Module-level imports: use `crate::FRAMES_PER_SECOND;` directly. Remove the `println!` line inside `note_detection_with_onset_offset_regress` (libraries must not print).

- [x] **Step 2: Shrink main.rs into a thin shell**

`neothesia-ai/src/main.rs` keeps `mod args; mod audio;` and `fn main` with an unchanged body, adding at the top:

```rust
use neothesia_ai::{SEGMENT_SAMPLES, deframe, enframe, get_binarized_output_from_regression, note_detection_with_onset_offset_regress};
```

If `audio.rs` still resolves `crate::{SAMPLE_RATE, SEGMENT_SAMPLES}` inside the same bin crate it keeps working as-is; if compilation fails, switch it to `use neothesia_ai::{...}` too.

- [x] **Step 3: Write the failing tests (lock binarization behavior)**

Append to `neothesia-ai/src/transcription.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn binarization_flags_threshold_peak() {
        // Single pitch, 5 frames: [0.0, 0.5, 0.9, 0.5, 0.0] peak at frame 2
        let mut data = Array2::<f32>::zeros((5, 1));
        for (i, v) in [0.0_f32, 0.5, 0.9, 0.5, 0.0] {
            data[[i, 0]] = v;
        }

        let (binary, _shift) =
            get_binarized_output_from_regression(&data.view(), 0.3, 1);

        assert!(binary[[2, 0]], "peak frame must be flagged as onset");
        assert!(!binary[[0, 0]] && !binary[[4, 0]], "edge frames must not be flagged");
    }

    #[test]
    fn binarization_rejects_flat_noise() {
        let mut data = Array2::<f32>::zeros((6, 1));
        for i in 0..6 {
            data[[i, 0]] = 0.05; // all below threshold
        }
        let (binary, _) = get_binarized_output_from_regression(&data.view(), 0.3, 1);
        assert!(binary.iter().all(|&b| !b));
    }
}
```

- [x] **Step 4: Run the tests**

Run: `cargo test -p neothesia-ai`
Expected: 2 passed (note: with neighbour=1 the valid n range is 1..4, which the test data satisfies).

- [x] **Step 5: Confirm the CLI still works**

Run: `cargo check -p neothesia-ai --bin neothesia-ai`
Expected: compiles without unused-import warnings.

- [x] **Step 6: Commit**

```bash
git add neothesia-ai
git commit -m "refactor(neothesia-ai): extract transcription core into library"
```

---

### Task 1.2: Create the audio-input crate scaffold

**Files:**
- Create: `audio-input/Cargo.toml`
- Create: `audio-input/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [x] **Step 1: Register in the workspace**

Add `"audio-input"` to the `members` array of the root `Cargo.toml`; add to `[workspace.dependencies]`:

```toml
audio-input = { path = "./audio-input" }
rubato = "0.16"
neothesia-ai = { path = "./neothesia-ai" }
rten = "0.24"
```

(`neothesia-ai` and `rten` are not workspace deps yet and are needed by audio-input later.)

- [x] **Step 2: Crate manifest**

`audio-input/Cargo.toml`:

```toml
[package]
name = "audio-input"
description = "Microphone audio input with polyphonic pitch detection for Neothesia"
version = "0.1.0"
edition.workspace = true

[dependencies]
log.workspace = true
thiserror.workspace = true
cpal.workspace = true
rubato.workspace = true
neothesia-ai.workspace = true
rten.workspace = true
```

- [x] **Step 3: Minimal lib.rs**

`audio-input/src/lib.rs`:

```rust
//! Microphone capture + streaming polyphonic pitch detection.
//!
//! Public API mirrors `midi_io`: enumerate devices, connect, receive
//! note events through a callback on a background thread.

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
```

(The `buffer` / `resample` module declarations are added by their own tasks — Task 1.3 / Task 1.4 — to keep each step self-contained.)

- [x] **Step 4: Verify compilation**

Run: `cargo check -p audio-input`
Expected: passes.

- [x] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock audio-input
git commit -m "feat(audio-input): scaffold workspace crate"
```

---

### Task 1.3: SPSC audio buffer

**Files:**
- Create: `audio-input/src/buffer.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod buffer;`)

Note: the design doc says "lock-free ring buffer". Implementation decision (recorded in the design doc appendix): a `Mutex<VecDeque<f32>>` with bounded SPSC semantics — the write side locks once per callback (~10-20ms), the read side once per 60ms hop, so lock contention is negligible; this avoids ringbuf third-party API risk. On overflow the oldest samples are dropped (design §5).

- [x] **Step 1: Write the failing tests together with the implementation**

`audio-input/src/buffer.rs`:

```rust
use std::collections::VecDeque;
use std::sync::Mutex;

/// Bounded SPSC sample buffer shared between the cpal callback
/// (producer) and the inference thread (consumer). On overflow the
/// oldest samples are dropped.
#[derive(Debug)]
pub struct SampleBuffer {
    inner: Mutex<VecDeque<f32>>,
    capacity: usize,
}

impl SampleBuffer {
    pub fn new(capacity_samples: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity_samples)),
            capacity: capacity_samples,
        }
    }

    /// Producer side. Never blocks longer than a mutex lock.
    pub fn push(&self, samples: &[f32]) {
        let mut q = self.inner.lock().unwrap();
        for &s in samples {
            if q.len() == self.capacity {
                q.pop_front();
            }
            q.push_back(s);
        }
    }

    /// Consumer side: drain everything currently buffered.
    pub fn drain(&self) -> Vec<f32> {
        let mut q = self.inner.lock().unwrap();
        q.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order() {
        let b = SampleBuffer::new(10);
        b.push(&[1.0, 2.0, 3.0]);
        assert_eq!(b.drain(), vec![1.0, 2.0, 3.0]);
        assert!(b.is_empty());
    }

    #[test]
    fn overflow_drops_oldest() {
        let b = SampleBuffer::new(3);
        b.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b.drain(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn concurrent_push_drain_does_not_deadlock() {
        let b = std::sync::Arc::new(SampleBuffer::new(48000));
        let producer = {
            let b = b.clone();
            std::thread::spawn(move || {
                let chunk = vec![0.5_f32; 480];
                for _ in 0..100 {
                    b.push(&chunk);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            })
        };
        for _ in 0..50 {
            let _ = b.drain();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        producer.join().unwrap();
    }
}
```

- [x] **Step 2: Run the tests**

Run: `cargo test -p audio-input`
Expected: 3 passed.

- [x] **Step 3: Expose the module and commit**

Add `pub mod buffer;` to `lib.rs`, then:

```bash
git add audio-input
git commit -m "feat(audio-input): bounded sample buffer with drop-oldest overflow"
```

---

### Task 1.4: Incremental resampler (arbitrary rate → 16kHz mono)

**Files:**
- Create: `audio-input/src/resample.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod resample;`)

- [x] **Step 1: Write the failing tests together with the implementation**

`audio-input/src/resample.rs`:

```rust
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use crate::TARGET_SAMPLE_RATE;

const CHUNK: usize = 1024;

/// Incremental resampler: push arbitrary amounts of mono f32 at
/// `source_rate`, pull 16 kHz mono f32 out. Internal leftover is
/// buffered until a full chunk is available.
pub struct ResampleStage {
    resampler: Option<SincFixedIn<f32>>,
    source_rate: u32,
    input_leftover: Vec<f32>,
}

impl ResampleStage {
    pub fn new(source_rate: u32) -> Self {
        let resampler = if source_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                oversampling_factor: 256,
                interpolation: SincInterpolationType::Linear,
                window: WindowFunction::BlackmanHarris2,
            };
            let ratio = TARGET_SAMPLE_RATE as f64 / source_rate as f64;
            Some(SincFixedIn::new(ratio, 1.2, params, CHUNK, 1).unwrap())
        };
        Self {
            resampler,
            source_rate,
            input_leftover: Vec::new(),
        }
    }

    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Feed raw samples; returns newly produced 16 kHz samples
    /// (possibly empty if not enough input for a full chunk yet).
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let Some(resampler) = self.resampler.as_mut() else {
            return input.to_vec(); // 16k passthrough
        };

        self.input_leftover.extend_from_slice(input);

        let mut produced = Vec::new();
        while self.input_leftover.len() >= CHUNK {
            let chunk: Vec<f32> = self.input_leftover.drain(..CHUNK).collect();
            let mut out = vec![vec![0.0_f32; resampler.output_frames_max()]];
            let (in_used, out_written) = resampler
                .process_into_buffer(&[chunk], &mut out, None)
                .expect("resample failed");
            debug_assert_eq!(in_used, CHUNK);
            produced.extend_from_slice(&out[0][..out_written]);
        }

        produced
    }

    /// Samples still buffered waiting for a full chunk (diagnostic).
    pub fn pending_input(&self) -> usize {
        self.input_leftover.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Estimate dominant frequency by zero crossings over 1s.
    fn dominant_freq(samples: &[f32], rate: u32) -> f32 {
        let mut crossings = 0;
        for w in samples.windows(2) {
            if (w[0] < 0.0 && w[1] >= 0.0) || (w[0] >= 0.0 && w[1] < 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / 2.0 * (rate as f32 / samples.len() as f32)
    }

    fn sine(freq: f32, rate: u32, seconds: f32) -> Vec<f32> {
        (0..(rate as f32 * seconds) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn preserves_440hz_from_48k() {
        let mut stage = ResampleStage::new(48_000);
        let input = sine(440.0, 48_000, 3.0);
        // feed in odd-sized chunks to simulate streaming
        let mut out = Vec::new();
        for chunk in input.chunks(777) {
            out.extend(stage.process(chunk));
        }
        // trim 0.25s head/tail to skip boundary effects
        let skip = TARGET_SAMPLE_RATE as usize / 4;
        let core = &out[skip..out.len() - skip];
        let f = dominant_freq(core, TARGET_SAMPLE_RATE);
        assert!(
            (f - 440.0).abs() < 2.0,
            "frequency drifted: {f} Hz"
        );
    }

    #[test]
    fn passthrough_when_already_16k() {
        let mut stage = ResampleStage::new(16_000);
        let out = stage.process(&[0.1, 0.2, 0.3]);
        assert_eq!(out, vec![0.1, 0.2, 0.3]);
    }
}
```

- [x] **Step 2: Run the tests**

Run: `cargo test -p audio-input`
Expected: all pass (including Task 1.3's 3). If rubato call names differ (version drift), fix the call sites per the 0.16 docs — but **do not change test semantics**.

- [x] **Step 3: Commit**

```bash
git add audio-input Cargo.lock
git commit -m "feat(audio-input): incremental sinc resampler to 16kHz mono"
```

---

### Task 1.5: Microphone device enumeration and capture stream

**Files:**
- Create: `audio-input/src/capture.rs`
- Create: `audio-input/examples/mic_list.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod capture;`)

- [x] **Step 1: Implement the capture module**

`audio-input/src/capture.rs`:

```rust
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::buffer::SampleBuffer;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("no input device named {0:?}")]
    DeviceNotFound(String),
    #[error("no input devices available")]
    NoDevices,
    #[error("failed to configure input stream: {0}")]
    Config(String),
    #[error(transparent)]
    Cpal(#[from] cpal::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MicDevice(pub String);

impl std::fmt::Display for MicDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub fn devices() -> Vec<MicDevice> {
    let host = cpal::default_host();
    host.input_devices()
        .filter_map(|d| d.name().ok())
        .map(MicDevice)
        .collect()
}

pub struct CaptureStream {
    pub device: MicDevice,
    pub sample_rate: u32,
    pub buffer: Arc<SampleBuffer>,
    /// Set by the cpal error callback (device unplugged etc.);
    /// polled by the inference thread.
    pub error: Arc<std::sync::atomic::AtomicBool>,
    _stream: cpal::Stream,
}

/// Open the default input config of `device`, convert to f32 mono,
/// and push into a shared SampleBuffer.
pub fn connect(device: &MicDevice) -> Result<CaptureStream, CaptureError> {
    let host = cpal::default_host();
    let dev = host
        .input_devices()?
        .find(|d| d.name().ok().as_deref() == Some(device.0.as_str()))
        .ok_or_else(|| CaptureError::DeviceNotFound(device.0.clone()))?;

    let config = dev
        .default_input_config()
        .map_err(|e| CaptureError::Config(e.to_string()))?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let buffer = Arc::new(SampleBuffer::new(sample_rate as usize * 4)); // ~4s
    let error = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let buf = buffer.clone();
    let err_flag = error.clone();
    let err_fn = move |e| {
        log::error!("audio input stream error: {e}");
        err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            &config.into(),
            move |data: &[f32], _| push_mono(&buf, data, channels),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream(
            &config.into(),
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|s| s as f32 / i16::MAX as f32).collect();
                push_mono(&buf, &f, channels);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => dev.build_input_stream(
            &config.into(),
            move |data: &[u16], _| {
                let f: Vec<f32> = data
                    .iter()
                    .map(|s| (*s as f32 - 32768.0) / 32768.0)
                    .collect();
                push_mono(&buf, &f, channels);
            },
            err_fn,
            None,
        )?,
        other => {
            return Err(CaptureError::Config(format!(
                "unsupported sample format: {other:?}"
            )))
        }
    };

    stream.play()?;

    Ok(CaptureStream {
        device: device.clone(),
        sample_rate,
        buffer,
        error,
        _stream: stream,
    })
}

fn push_mono(buf: &SampleBuffer, interleaved: &[f32], channels: usize) {
    if channels <= 1 {
        buf.push(interleaved);
        return;
    }
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect();
    buf.push(&mono);
}
```

- [x] **Step 2: examples/mic_list.rs (manual smoke tool)**

```rust
fn main() {
    println!("Input devices:");
    for d in audio_input::capture::devices() {
        println!("  - {d}");
    }
}
```

- [x] **Step 3: Compile + manual smoke test**

Run: `cargo run -p audio-input --example mic_list`
Expected: lists the machine's input devices (on macOS the first run may trigger a permission prompt; a bare terminal example without Info.plist may be denied outright — either allow it in System Settings or treat compile-only success as sufficient for this step).

- [x] **Step 4: Commit**

```bash
git add audio-input
git commit -m "feat(audio-input): mic device enumeration and capture stream"
```

---

## Phase 1 completion record

- Task 1.1 — commit ca25082 (+ fmt 19303b6): verbatim move verified; println/labels removed; float-cast fixes
- Task 1.2 — commit 88b456c: rten already existed in workspace deps (plan text was wrong), duplicate correctly skipped
- Task 1.3 — commits 2e3ef35, c551872: poisoning-tolerant locks + zero-capacity guard added post-review
- Task 1.4 — commits 8dfe7c3, c225451: rubato 0.16.2 API matched verbatim; 44.1kHz + output-length tests added post-review
- Task 1.5 — commits ee700b2, dffc47c: cpal 0.18.1 adaptations (Display for names, Result input_devices, StreamConfig by value, per-arm error closures); push_mono tests
