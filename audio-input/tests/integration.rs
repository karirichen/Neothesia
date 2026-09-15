//! Offline detection regression. Requires the real model:
//! NTS_TEST_MODEL=<path> cargo test -p audio-input --test integration -- --ignored --nocapture

use audio_input::detector::rten_backend::RtenDetector;
use audio_input::pipeline::StreamingPipeline;
use audio_input::tracker::{MicEvent, TrackerConfig};
use audio_input::{HOP_SAMPLES, TARGET_SAMPLE_RATE};

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
        fed += chunk.len();
        for ev in p.push(chunk) {
            if matches!(ev, MicEvent::NoteOn { .. }) {
                onsets_at.push(fed);
            }
            events.push(ev);
        }
    }
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
    // The note must start AFTER the first window: the pipeline cannot
    // run inference before 1.5s (WINDOW_SAMPLES) of audio exists, so a
    // strike at 0.5s is reported at the 1.5s mark — 1000ms of cold-start
    // latency by construction, not model regression. Steady state is
    // trust margin (120ms) + hop quantization (<=60ms); that is what
    // this budget guards.
    let start = (2.0 * SAMPLE_RATE as f32) as usize;
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
