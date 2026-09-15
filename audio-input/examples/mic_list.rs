fn main() {
    println!("Input devices:");
    for d in audio_input::capture::devices() {
        println!("  - {d}");
    }
}
