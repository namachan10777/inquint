use clap::Parser;
use quintck::explorer::{check, CheckConfig, CheckOutcome};
use quintck::spec::{CompiledSpec, EntryPoints};
use std::path::PathBuf;
use std::process::ExitCode;

mod compile;

/// quintck — explicit-state (TLC-like) model checker for Quint.
///
/// Accepts a .qnt file (compiled via the `quint` CLI, which must be on
/// PATH) or a pre-compiled .json file (`quint compile --target=json`).
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Input spec: .qnt or compiled .json
    input: PathBuf,

    /// Name of the main module (forwarded to quint compile)
    #[arg(long)]
    main: Option<String>,

    /// Invariants to check, comma-separated
    #[arg(long, value_delimiter = ',')]
    invariant: Vec<String>,

    /// Name of the initializer action [default: q::init / init]
    #[arg(long)]
    init: Option<String>,

    /// Name of the step action [default: q::step / step]
    #[arg(long)]
    step: Option<String>,

    /// Maximum number of steps from an initial state
    #[arg(long, default_value_t = 10)]
    max_steps: u32,

    /// Explore the full state space regardless of depth
    #[arg(long)]
    exhaustive: bool,

    /// Do not report deadlocks (states with no successor)
    #[arg(long)]
    no_deadlock: bool,

    /// Abort (without a verdict) after this many states
    #[arg(long)]
    max_states: Option<u64>,

    /// Write the counterexample trace in ITF format to this file
    #[arg(long)]
    out_itf: Option<PathBuf>,

    /// Temporal properties (not supported in v1; fails immediately)
    #[arg(long)]
    temporal: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if cli.temporal.is_some() {
        eprintln!("error: temporal properties are not supported by quintck v1");
        return ExitCode::from(2);
    }

    let opts = compile::QntOptions {
        main: cli.main.as_deref(),
        init: cli.init.as_deref(),
        step: cli.step.as_deref(),
        invariants: &cli.invariant,
    };
    let json = match compile::load_input(&cli.input, &opts) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let output = match quint_ast::CompiledOutput::load(&json) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let entry = EntryPoints {
        init: cli.init,
        step: cli.step,
        invariants: cli.invariant,
    };

    let spec = match CompiledSpec::build(&output, &entry) {
        Ok(spec) => spec,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let cfg = CheckConfig {
        max_steps: if cli.exhaustive {
            None
        } else {
            Some(cli.max_steps)
        },
        deadlock: !cli.no_deadlock,
        max_states: cli.max_states,
    };

    let source = cli.input.display().to_string();
    let write_itf = |trace: &[quintck::state::State], violation: bool| {
        if let Some(path) = &cli.out_itf {
            let itf = quintck::itf_out::trace_to_itf(&spec.vars.names, trace, violation, &source);
            match serde_json::to_string_pretty(&itf) {
                Ok(json) => {
                    if let Err(e) = std::fs::write(path, json) {
                        eprintln!("warning: cannot write {}: {e}", path.display());
                    }
                }
                Err(e) => eprintln!("warning: cannot serialize ITF trace: {e}"),
            }
        }
    };

    let print_trace = |trace: &[quintck::state::State]| {
        for (i, state) in trace.iter().enumerate() {
            eprintln!("State {i}:");
            for (name, value) in spec.vars.names.iter().zip(state.iter()) {
                eprintln!("  {name} = {value}");
            }
        }
    };

    match check(&spec, &cfg) {
        Ok(CheckOutcome::Pass { states, max_depth }) => {
            println!("[ok] {states} states explored, depth <= {max_depth}, invariants hold");
            ExitCode::SUCCESS
        }
        Ok(CheckOutcome::InvariantViolation { invariant, trace }) => {
            print_trace(&trace);
            write_itf(&trace, true);
            println!(
                "[violation] invariant '{invariant}' violated at depth {}",
                trace.len() - 1
            );
            ExitCode::from(1)
        }
        Ok(CheckOutcome::Deadlock { trace }) => {
            print_trace(&trace);
            write_itf(&trace, true);
            println!("[violation] deadlock reached at depth {}", trace.len() - 1);
            ExitCode::from(1)
        }
        Ok(CheckOutcome::Incomplete { states }) => {
            eprintln!("stopped after {states} states (max-states); no verdict");
            ExitCode::from(2)
        }
        Err(e) => {
            if !e.trace.is_empty() {
                print_trace(&e.trace);
            }
            eprintln!("error: {}", e.error);
            ExitCode::from(2)
        }
    }
}
