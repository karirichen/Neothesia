//! Standalone mic pitch-detection tester.
//!
//!   RUST_LOG=audio_input=info cargo run --release -p audio-input --example mic_detect
//!
//! Play notes on the piano; each detection prints the note name. A
//! heartbeat line every 5s shows the mic level (RMS) and event count —
//! rms≈0 means no audio is reaching the mic (wrong device / permission),
//! healthy rms with 0 events means detection (not capture) trouble.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use audio_input::{AudioInputManager, MicDevice, MicEvent};

fn note_name(midi: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", NAMES[(midi as usize) % 12], midi as i16 / 12 - 1)
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    let devices = AudioInputManager::devices();
    println!("== Mic pitch-detection tester ==");
    println!("Devices:");
    for (i, d) in devices.iter().enumerate() {
        println!("  [{i}] {d}");
    }

    let device = if filter.is_empty() {
        MicDevice(
            AudioInputManager::default_device()
                .map(|d| d.0)
                .unwrap_or_else(|| {
                    devices
                        .first()
                        .cloned()
                        .map(|d| d.0)
                        .expect("no mic devices")
                }),
        )
    } else {
        let found = devices.iter().find(|d| d.0.contains(&filter));
        match found {
            Some(d) => d.clone(),
            None => {
                println!("No device matches '{filter}'");
                return;
            }
        }
    };
    println!("\nUsing device: {device}");
    println!("Play notes on the piano (Ctrl+C to stop). Expected latency ~0.3-0.5s.\n");

    let model_path = audio_input::model_store::model_path();
    println!(
        "Model: {} (exists: {})\n",
        model_path.display(),
        model_path.exists()
    );

    let done = Arc::new(AtomicBool::new(false));
    let done2 = done.clone();
    let conn = match AudioInputManager::connect(&device, &model_path, move |ev| match ev {
        MicEvent::NoteOn { key } => println!("  NoteOn   {} (midi {key})", note_name(key)),
        MicEvent::NoteOff { key } => println!("  NoteOff  {} (midi {key})", note_name(key)),
        MicEvent::Error(msg) => {
            println!("  ERROR: {msg}");
            done2.store(true, Ordering::Relaxed);
        }
    }) {
        Ok(c) => c,
        Err(e) => {
            println!("connect failed: {e}");
            return;
        }
    };

    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) && !done.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
    }
    drop(conn);
    println!("\nDone.");
}
