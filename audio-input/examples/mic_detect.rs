use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use audio_input::{AudioInputManager, MicDevice, MicEvent};

fn main() {
    let device_name = std::env::args().nth(1).unwrap_or_default();
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let devices = AudioInputManager::devices();
    println!("Devices: {devices:?}");
    let device = if device_name.is_empty() {
        MicDevice(
            devices
                .iter()
                .find(|d| d.0.contains("MacBook"))
                .cloned()
                .map(|d| d.0)
                .unwrap_or_else(|| devices[0].0.clone()),
        )
    } else {
        MicDevice(
            devices
                .iter()
                .find(|d| d.0 == device_name)
                .map(|d| d.0.clone())
                .unwrap_or(device_name),
        )
    };
    println!("Connecting to: {device}");

    let model_path = audio_input::model_store::model_path();
    println!(
        "Model: {} (exists: {})",
        model_path.display(),
        model_path.exists()
    );

    let done = Arc::new(AtomicBool::new(false));
    let done2 = done.clone();
    let conn = AudioInputManager::connect(&device, &model_path, move |ev| match ev {
        MicEvent::NoteOn { key } => println!("  NoteOn  {key}"),
        MicEvent::NoteOff { key } => println!("  NoteOff {key}"),
        MicEvent::Error(msg) => {
            println!("  ERROR: {msg}");
            done2.store(true, Ordering::Relaxed);
        }
    })
    .map_err(|e| e.to_string())
    .expect("connect failed");

    println!("Connected. Listening for {seconds}s (play audio near the mic)...");
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) && !done.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
    }
    drop(conn);
    println!("Done.");
}
