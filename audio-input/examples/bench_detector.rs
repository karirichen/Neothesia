use std::time::Instant;

use audio_input::detector::PitchDetector;
use audio_input::detector::rten_backend::RtenDetector;
use audio_input::{TARGET_SAMPLE_RATE, WINDOW_SAMPLES};

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench <model.rten>");
    let t0 = Instant::now();
    let mut det = RtenDetector::load(path.as_ref()).unwrap();
    println!("load: {:?}", t0.elapsed());

    let rng_seed = 42u64;
    let mut s = rng_seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64 - 0.5) as f32 * 0.2
    };
    let window: Vec<f32> = (0..WINDOW_SAMPLES).map(|_| next()).collect();

    // warmup
    let t1 = Instant::now();
    let _ = det.detect(&window, 0);
    println!("first detect (cold): {:?}", t1.elapsed());

    for i in 0..5 {
        let t = Instant::now();
        let probs = det.detect(&window, i * 150);
        println!("detect {}: {:?} (frames={})", i, t.elapsed(), probs.frames);
    }

    // a realistic struck note: 440Hz-ish tone with harmonics, ~0.3 gain
    let mut tone = vec![0.0_f32; WINDOW_SAMPLES];
    let start = TARGET_SAMPLE_RATE as usize / 2;
    for i in start..WINDOW_SAMPLES {
        let t = (i - start) as f32 / TARGET_SAMPLE_RATE as f32;
        let env = (-t * 1.5).exp() * (1.0 - (-t * 50.0).exp());
        let f = 440.0_f32;
        tone[i] = env
            * 0.3
            * ((2.0 * std::f32::consts::PI * f * t).sin()
                + 0.4 * (2.0 * std::f32::consts::PI * 2.0 * f * t).sin());
    }
    let t = Instant::now();
    let probs = det.detect(&tone, 0);
    println!(
        "tone detect: {:?}, onset cols flagged: {}",
        t.elapsed(),
        probs.onset.iter().filter(|&&o| o).count()
    );
    let key69: Vec<usize> = (0..probs.frames)
        .filter(|&f| probs.onset_at(f, 69 - 21))
        .collect();
    println!("pitch 69 onset frames: {key69:?}");
}
