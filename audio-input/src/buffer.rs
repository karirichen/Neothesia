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
