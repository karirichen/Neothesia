use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
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
        assert!((f - 440.0).abs() < 2.0, "frequency drifted: {f} Hz");
    }

    #[test]
    fn passthrough_when_already_16k() {
        let mut stage = ResampleStage::new(16_000);
        let out = stage.process(&[0.1, 0.2, 0.3]);
        assert_eq!(out, vec![0.1, 0.2, 0.3]);
    }
}
