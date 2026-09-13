//! First-use download + on-disk cache of the pitch detection model.

use std::io::Read;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Points at this fork's GitHub release attachment.
/// Phase 5 Task 5.1 backfills URL + SHA256 after the model is uploaded.
pub const MODEL_URL: &str =
    "https://github.com/karirichen/Neothesia/releases/download/models/piano_transcription.rten";
pub const MODEL_SHA256: &str = "REPLACE_WITH_ACTUAL_SHA256_AT_RELEASE";

const MODEL_FILE: &str = "piano_transcription.rten";

#[derive(Debug, thiserror::Error)]
pub enum ModelStoreError {
    #[error("download failed: {0}")]
    Download(String),
    #[error("checksum mismatch for {path:?}")]
    Checksum { path: PathBuf },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub fn model_path() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("neothesia")
        .join("models")
        .join(MODEL_FILE)
}

/// Ensure the model exists locally (download if missing), verify
/// checksum, return its path.
pub fn ensure_model() -> Result<PathBuf, ModelStoreError> {
    let path = model_path();
    if path.exists() && verify_sha256(&path, MODEL_SHA256)? {
        return Ok(path);
    }
    // corrupted or missing → re-download
    if path.exists() {
        std::fs::remove_file(&path)?;
    }

    let response = ureq::get(MODEL_URL)
        .call()
        .map_err(|e| ModelStoreError::Download(e.to_string()))?;

    let bytes: Vec<u8> = std::io::BufReader::new(response.into_reader())
        .take(64 * 1024 * 1024) // hard cap 64MB
        .bytes()
        .collect::<Result<_, _>>()?;

    if path.parent().is_some() {
        std::fs::create_dir_all(path.parent().unwrap())?;
    }
    std::fs::write(&path, &bytes)?;

    if !verify_sha256(&path, MODEL_SHA256)? {
        return Err(ModelStoreError::Checksum { path });
    }

    Ok(path)
}

fn verify_sha256(path: &std::path::Path, expected_hex: &str) -> Result<bool, ModelStoreError> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = hex_lower(&hasher.finalize());
    Ok(constant_time_eq(actual.as_bytes(), expected_hex.as_bytes()))
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Trivial constant-time comparison to avoid timing oracles
/// (defensive; the threat model here is minimal).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_known_bytes() {
        let dir = std::env::temp_dir().join("nts-model-test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("known.bin");
        std::fs::write(&f, b"hello world").unwrap();
        // sha256("hello world")
        assert!(
            verify_sha256(
                &f,
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            )
            .unwrap()
        );
        assert!(!verify_sha256(&f, "deadbeef").unwrap());
    }
}
