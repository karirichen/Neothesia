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
        .map(|devices| devices.map(|d| MicDevice(d.to_string())).collect())
        .unwrap_or_else(|e| {
            log::warn!("failed to enumerate input devices: {e}");
            Vec::new()
        })
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

fn err_handler(
    flag: Arc<std::sync::atomic::AtomicBool>,
) -> impl FnMut(cpal::Error) + Send + 'static {
    move |e| {
        log::error!("audio input stream error: {e}");
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Open the default input config of `device`, convert to f32 mono,
/// and push into a shared SampleBuffer.
pub fn connect(device: &MicDevice) -> Result<CaptureStream, CaptureError> {
    let host = cpal::default_host();
    let dev = host
        .input_devices()?
        .find(|d| d.to_string() == device.0)
        .ok_or_else(|| CaptureError::DeviceNotFound(device.0.clone()))?;

    let config = dev
        .default_input_config()
        .map_err(|e| CaptureError::Config(e.to_string()))?;

    let format = config.sample_format();
    let sample_rate = config.sample_rate();
    let channels = config.channels() as usize;
    let stream_config: cpal::StreamConfig = config.into();
    let buffer = Arc::new(SampleBuffer::new(sample_rate as usize * 4)); // ~4s
    let error = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let buf = buffer.clone();
    let err_flag = error.clone();
    let stream = match format {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            stream_config,
            move |data: &[f32], _| push_mono(&buf, data, channels),
            err_handler(err_flag.clone()),
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream(
            stream_config,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                push_mono(&buf, &f, channels);
            },
            err_handler(err_flag.clone()),
            None,
        )?,
        cpal::SampleFormat::U16 => dev.build_input_stream(
            stream_config,
            move |data: &[u16], _| {
                let f: Vec<f32> = data
                    .iter()
                    .map(|s| (*s as f32 - 32768.0) / 32768.0)
                    .collect();
                push_mono(&buf, &f, channels);
            },
            err_handler(err_flag),
            None,
        )?,
        other => {
            return Err(CaptureError::Config(format!(
                "unsupported sample format: {other:?}"
            )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::SampleBuffer;

    #[test]
    fn mono_passthrough() {
        let b = SampleBuffer::new(16);
        push_mono(&b, &[0.1, 0.2, 0.3], 1);
        assert_eq!(b.drain(), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn stereo_average() {
        let b = SampleBuffer::new(16);
        push_mono(&b, &[0.0, 0.4, 1.0, 0.2], 2);
        assert_eq!(b.drain(), vec![0.2, 0.6]);
    }

    #[test]
    fn trailing_partial_frame_averages_by_channel_count() {
        // A trailing frame with fewer samples than channels divides by
        // the full channel count (attenuated, never panics) — pinned
        // deliberately: cpal delivers complete interleaved frames, so
        // this documents the degradation path only.
        let b = SampleBuffer::new(16);
        push_mono(&b, &[0.0, 0.4, 1.0], 2);
        assert_eq!(b.drain(), vec![0.2, 0.5]); // 1.0 / 2
    }
}
