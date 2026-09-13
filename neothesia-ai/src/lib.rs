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
