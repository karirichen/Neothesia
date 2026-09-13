//! Microphone capture + streaming polyphonic pitch detection.
//!
//! Public API mirrors `midi_io`: enumerate devices, connect, receive
//! note events through a callback on a background thread.

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
