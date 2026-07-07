//! Input loading: pre-compiled JSON directly, or .qnt via `quint compile`.

use std::path::Path;
use std::process::{Command, Stdio};

pub struct QntOptions<'a> {
    pub main: Option<&'a str>,
    pub init: Option<&'a str>,
    pub step: Option<&'a str>,
    pub invariants: &'a [String],
    pub temporal: &'a [String],
}

pub fn load_input(input: &Path, opts: &QntOptions) -> Result<String, String> {
    match input.extension().and_then(|e| e.to_str()) {
        Some("json") => std::fs::read_to_string(input)
            .map_err(|e| format!("cannot read {}: {e}", input.display())),
        Some("qnt") => {
            // The flattened compilation target is written to stdout only
            // (`--out` would give the pre-flattening stage). Reading a node
            // process's stdout through a pipe silently truncates at 64KiB
            // when node exits with pending async writes, so redirect stdout
            // into a temp file instead — file writes are synchronous.
            let tmp = std::env::temp_dir().join(format!(
                "quintck-compile-{}.json",
                std::process::id()
            ));
            let stdout_file = std::fs::File::create(&tmp)
                .map_err(|e| format!("cannot create temp file {}: {e}", tmp.display()))?;

            let mut cmd = Command::new("quint");
            cmd.arg("compile")
                .arg("--target=json")
                .arg(input)
                .stdout(Stdio::from(stdout_file));
            if let Some(main) = opts.main {
                cmd.arg(format!("--main={main}"));
            }
            if let Some(init) = opts.init {
                cmd.arg(format!("--init={init}"));
            }
            if let Some(step) = opts.step {
                cmd.arg(format!("--step={step}"));
            }
            if !opts.invariants.is_empty() {
                cmd.arg(format!("--invariant={}", opts.invariants.join(",")));
            }
            if !opts.temporal.is_empty() {
                cmd.arg(format!("--temporal={}", opts.temporal.join(",")));
            }

            let output = cmd
                .output()
                .map_err(|e| format!("cannot run `quint compile` (is quint on PATH?): {e}"))?;
            if !output.status.success() {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!(
                    "quint compile failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            let json = std::fs::read_to_string(&tmp)
                .map_err(|e| format!("cannot read quint compile output {}: {e}", tmp.display()));
            let _ = std::fs::remove_file(&tmp);
            json
        }
        _ => Err(format!(
            "unsupported input {}: expected .qnt or .json",
            input.display()
        )),
    }
}
