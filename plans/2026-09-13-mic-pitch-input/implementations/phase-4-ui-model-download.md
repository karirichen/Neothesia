# Phase 4: Settings UI + Config Persistence + Model Download — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Microphone section on the settings page (toggle + device selection), config persistence, first-use model download and caching.

**Architecture:** The config follows the existing `V1 struct + enum wrapper + #[serde(default)]` pattern with a new `mic` section; the downloader lives in the audio-input crate (`model_store`); the UI uses nuon's existing `settings_row_toggler`/`settings_row_spin` patterns with the async download going through the existing `on_async` + `MenuScene::futures` mechanism.

**Tech Stack:** nuon UI, ureq 2 (blocking HTTP), sha2, dirs, existing ron config.

**Design doc:** `plans/2026-09-13-mic-pitch-input/design.md` §7, §8

---

### Task 4.1: Config model gains a mic section

**Files:**
- Modify: `neothesia-core/src/config/model.rs`
- Modify: `neothesia-core/src/config/mod.rs`

- [ ] **Step 1: Add the types to model.rs**

Append after the `PcKeyboardConfig` definitions (keeping the file's existing style):

```rust
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct MicConfigV1 {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub device: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub enum MicConfig {
    V1(MicConfigV1),
}

impl Default for MicConfig {
    fn default() -> Self {
        Self::V1(MicConfigV1::default())
    }
}
```

The top-level `Model` (model.rs:7) gains:

```rust
    #[serde(default)]
    pub mic: MicConfig,
```

(`Model` carries `deny_unknown_fields`; `#[serde(default)]` guarantees configs without a `mic` key still parse.)

- [ ] **Step 2: Accessors in mod.rs**

Next to `set_separate_channels`/`separate_channels` (around config/mod.rs:127-132), append:

```rust
    pub fn mic_enabled(&self) -> bool {
        self.mic.enabled
    }

    pub fn set_mic_enabled(&mut self, enabled: bool) {
        self.mic.enabled = enabled;
    }

    pub fn mic_device(&self) -> Option<&str> {
        self.mic.device.as_deref()
    }

    pub fn set_mic_device(&mut self, device: Option<String>) {
        self.mic.device = device;
    }
```

Note: mod.rs stores the model's fields unpacked (e.g. `devices: DevicesConfigV1`, unpacked from the enum around mod.rs:73/97). If `mic` needs the same unpacking, add an internal `mic: MicConfigV1` field plus the unpack line `mic: match model.mic { MicConfig::V1(v) => v }` and the repack `MicConfig::V1(self.mic.clone())` — follow the on-site structure; the goal is: the accessors compile and old ron config files load without errors.

- [ ] **Step 3: Regression**

Run: `cargo test -p neothesia-core`
Expected: existing tests pass (nothing broken). If a config roundtrip test exists, run it.

- [ ] **Step 4: Commit**

```bash
git add neothesia-core
git commit -m "feat(config): mic input settings section"
```

---

### Task 4.2: Model download and cache (model_store)

**Files:**
- Create: `audio-input/src/model_store.rs`
- Modify: `audio-input/src/lib.rs` (add `pub mod model_store;`)
- Modify: `audio-input/Cargo.toml`, root `Cargo.toml`

- [ ] **Step 1: Dependencies**

Add to the root `[workspace.dependencies]`:

```toml
ureq = { version = "2", default-features = false, features = ["tls"] }
sha2 = "0.10"
dirs = "6"
```

Add to `audio-input/Cargo.toml` `[dependencies]`:

```toml
ureq.workspace = true
sha2.workspace = true
dirs.workspace = true
```

- [ ] **Step 2: Implementation**

`audio-input/src/model_store.rs`:

```rust
//! First-use download + on-disk cache of the pitch detection model.

use std::io::Read;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Points at this fork's GitHub release attachment.
/// Task 5.1 backfills URL + SHA256 after the model is uploaded.
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

    let bytes: Vec<u8> = response
        .into_reader()
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
        assert!(verify_sha256(
            &f,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
        .unwrap());
        assert!(!verify_sha256(&f, "deadbeef").unwrap());
    }
}
```

The `REPLACE_WITH_ACTUAL_SHA256_AT_RELEASE` value in `MODEL_SHA256` is a **deliberate placeholder with a dedicated backfill task** (Phase 5 Task 5.1: upload the model → `shasum -a 256` → write it back). Until backfilled, `ensure_model`'s verification necessarily fails — so the Task 4.3 UI flow only completes once the model is in place. This is expected.

- [ ] **Step 3: Tests**

Run: `cargo test -p audio-input`
Expected: all pass (including the new sha256 test).

- [ ] **Step 4: Commit**

```bash
git add audio-input Cargo.toml Cargo.lock
git commit -m "feat(audio-input): model download and verified cache"
```

---

### Task 4.3: Settings page Microphone section

**Files:**
- Modify: `neothesia/src/scene/menu_scene/settings.rs`
- Modify: `neothesia/src/scene/menu_scene/state.rs` (UiState gains a download-state field)
- Modify: `neothesia/src/context.rs` (connect/disconnect helpers)

- [ ] **Step 1: State in state.rs**

Add to `UiState` in state.rs:

```rust
pub mic_setup: MicSetupState,
```

with the type:

```rust
#[derive(Debug, Clone, Default)]
pub enum MicSetupState {
    #[default]
    Idle,
    Downloading,
    Failed(String),
}
```

(Ensure `Default` derives correctly.)

- [ ] **Step 2: Connect helpers in context.rs**

```rust
impl Context {
    /// Establish/re-establish the microphone input connection.
    /// The caller must ensure the model is in place.
    pub fn connect_audio_input(&mut self) -> Result<(), String> {
        self.audio_input = None; // drop the old connection

        let device_name = self
            .config
            .mic_device()
            .map(str::to_owned)
            .or_else(|| {
                audio_input::AudioInputManager::devices()
                    .first()
                    .cloned()
                    .map(|d| d.to_string())
            });

        let Some(device_name) = device_name else {
            return Err("no microphone devices found".into());
        };

        let proxy = self.proxy.clone();
        let device = audio_input::MicDevice(device_name.clone());
        let model_path = audio_input::model_store::model_path();

        let conn = audio_input::AudioInputManager::connect(
            &device,
            &model_path,
            move |event| {
                if let Some(ev) = crate::mic_event_to_neothesia(event) {
                    proxy.send_event(ev).ok();
                }
            },
        )
        .map_err(|e| e.to_string())?;

        self.audio_input = Some(conn);
        Ok(())
    }

    pub fn disconnect_audio_input(&mut self) {
        self.audio_input = None;
    }
}
```

(`AudioInputConnection::drop` already stops the inference thread — implemented in Phase 2 Task 2.5.)

- [ ] **Step 3: Settings UI section**

Insert after the "Input" section (settings.rs:69-73):

```rust
                nuon::settings_section("Microphone Input")
                    .width(body_w)
                    .build(ui, |ui, rows, spacer| {
                        self.settings_mic_section(ctx, ui, rows, spacer);
                    });
```

Inside the `impl super::MenuScene` block (next to `settings_input_section`, around settings.rs:347), add the method:

```rust
    fn settings_mic_section(
        &mut self,
        ctx: &mut Context,
        ui: &mut nuon::Ui,
        rows: &dyn Fn(&mut nuon::Ui, nuon::SettingsRow<'_>),
        spacer: &dyn Fn(&mut nuon::Ui),
    ) {
        let enabled = ctx.config.mic_enabled();

        match &self.state.mic_setup {
            MicSetupState::Downloading => {
                nuon::settings_row()
                    .title("Microphone")
                    .subtitle("Downloading pitch model...")
                    .build(ui, rows);
                return; // no toggling while downloading
            }
            MicSetupState::Failed(msg) => {
                nuon::settings_row()
                    .title("Microphone")
                    .subtitle(format!("Error: {msg}"))
                    .build(ui, rows);
            }
            _ => {}
        }

        if nuon::settings_row_toggler()
            .title("Enable Microphone Input")
            .subtitle("Detect played notes via mic (acoustic pianos)")
            .value(enabled)
            .build(ui, rows)
        {
            let target = !enabled;
            ctx.config.set_mic_enabled(target);
            if target {
                self.state.mic_setup = MicSetupState::Downloading;
                // on_async closure signature: (T, &mut UiState, &mut Context)
                let fut = async move {
                    audio_input::model_store::ensure_model()
                };
                let task = on_async(fut, |result, data, ctx| {
                    match result {
                        Ok(_path) => match ctx.connect_audio_input() {
                            Ok(()) => {
                                data.mic_setup = MicSetupState::Idle;
                            }
                            Err(e) => {
                                ctx.config.set_mic_enabled(false);
                                data.mic_setup = MicSetupState::Failed(e);
                            }
                        },
                        Err(e) => {
                            ctx.config.set_mic_enabled(false);
                            data.mic_setup = MicSetupState::Failed(e.to_string());
                        }
                    }
                });
                self.futures.push(task);
            } else {
                ctx.disconnect_audio_input();
                self.state.mic_setup = MicSetupState::Idle;
            }
        }

        spacer(ui);

        // device cycling (prompt when no devices)
        let devices: Vec<String> = audio_input::AudioInputManager::devices()
            .into_iter()
            .map(|d| d.to_string())
            .collect();

        if devices.is_empty() {
            nuon::settings_row()
                .title("Device")
                .subtitle("No microphone found — check Privacy & Security > Microphone")
                .build(ui, rows);
        } else {
            let current = ctx
                .config
                .mic_device()
                .map(str::to_owned)
                .unwrap_or_else(|| devices[0].clone());
            let spin = nuon::settings_row_spin()
                .title("Device")
                .subtitle(current.clone())
                .id("mic-device")
                .build(ui, rows);

            let idx = devices
                .iter()
                .position(|d| d == &current)
                .unwrap_or(0);
            let next = |dir: i32| {
                let n = devices.len() as i32;
                let ni = ((idx as i32 + dir + n) % n) as usize;
                devices[ni].clone()
            };

            let picked = match spin {
                nuon::SettingsRowSpinResult::Plus => Some(next(1)),
                nuon::SettingsRowSpinResult::Minus => Some(next(-1)),
                nuon::SettingsRowSpinResult::Idle => None,
            };
            if let Some(d) = picked {
                ctx.config.set_mic_device(Some(d));
                if ctx.config.mic_enabled() {
                    let _ = ctx.connect_audio_input();
                }
            }
        }
    }
```

Add to the top-level use statements: `use super::state::MicSetupState;` and (if not already imported) `crate::context::Context`.

Notes:
- `on_async` is a private fn at `menu_scene/mod.rs:31` returning `BoxFuture<MsgFn>`; the returned future must be registered via `self.futures.push(...)` (see `open_soundfont_picker` at settings.rs:509-517; futures are polled in `MenuScene::update`, mod.rs:313). If `on_async` is not visible from settings.rs (privacy), make it `pub(super)` or add a `pub fn spawn` wrapper in mod.rs — follow the compiler.
- `SettingsRowSpinResult::{Plus, Minus, Idle}` matches `update_range_start` (settings.rs:477-489).

- [ ] **Step 4: Restore the connection at startup**

In `Neothesia::new` in `main.rs` (or the initialization point right after Context construction):

```rust
    if context.config.mic_enabled()
        && audio_input::model_store::model_path().exists()
    {
        if let Err(e) = context.connect_audio_input() {
            log::warn!("mic input restore failed: {e}");
            context.config.set_mic_enabled(false);
        }
    }
```

(No automatic download at startup — the user must enable once via the settings page.)

- [ ] **Step 5: Compile + manual test**

Run: `cargo check --workspace && cargo run -p neothesia`
Manual checks: the Microphone Input section appears on the settings page; enabling it shows "Error: download failed..." until the model URL is backfilled (expected); the toggle and device selection persist into the ron config (verify by restarting).

- [ ] **Step 6: Commit**

```bash
git add neothesia audio-input
git commit -m "feat(ui): microphone input settings section with model download"
```
