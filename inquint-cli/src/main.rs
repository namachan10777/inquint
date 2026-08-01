use clap::Parser;
use inquint::explorer::{check, Assurance, CheckConfig, CheckOutcome, SearchScope};
use inquint::spec::{CompiledSpec, EntryPoints};
use std::path::PathBuf;
use std::process::ExitCode;

mod compile;

/// inquint — explicit-state (TLC-like) model checker for Quint.
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

    /// Keep full states for exact deduplication. By default the seen-set
    /// holds 64-bit fingerprints only (TLC-style): far less memory, with a
    /// negligible probability of missing a state on a fingerprint collision.
    #[arg(long)]
    exact_states: bool,

    /// Worker threads for invariant/deadlock checking (default: all
    /// cores). Temporal checking, run tests and --exact-states are
    /// single-threaded regardless. With --max-states, the parallel run
    /// may overshoot the cap by a few work chunks.
    #[arg(long)]
    threads: Option<usize>,

    /// UNSOUND measurement mode: explore with an optimistic
    /// partial-order reduction and print the reduced state count
    /// (granularity: "var" or "elem"). Verdicts are not trustworthy.
    #[arg(long, value_name = "GRANULARITY")]
    por_probe: Option<String>,

    /// Write the counterexample trace in ITF format to this file
    #[arg(long)]
    out_itf: Option<PathBuf>,

    /// Temporal properties to check, comma-separated
    #[arg(long, value_delimiter = ',')]
    temporal: Vec<String>,

    /// Execute `run` test definitions instead of model checking.
    /// Optionally pass test names; all runs execute by default.
    /// Every path through a run's nondeterminism must pass.
    #[arg(long, num_args = 0.., value_delimiter = ',')]
    test: Option<Vec<String>>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let opts = compile::QntOptions {
        main: cli.main.as_deref(),
        init: cli.init.as_deref(),
        step: cli.step.as_deref(),
        invariants: &cli.invariant,
        temporal: &cli.temporal,
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

    if let Some(test_names) = &cli.test {
        return match inquint::runner::run_tests(&output, test_names) {
            Ok(reports) => {
                let mut failed = 0;
                for report in &reports {
                    match &report.result {
                        Ok(paths) => println!("    ok {} ({paths} paths)", report.name),
                        Err(e) => {
                            println!("    failed {}: {e}", report.name);
                            failed += 1;
                        }
                    }
                }
                if failed == 0 {
                    println!("[ok] {} test(s) passed", reports.len());
                    ExitCode::SUCCESS
                } else {
                    println!("[violation] {failed} of {} test(s) failed", reports.len());
                    ExitCode::from(1)
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        };
    }

    let entry = EntryPoints {
        init: cli.init,
        step: cli.step,
        invariants: cli.invariant,
        temporal: cli.temporal.clone(),
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
        exact_states: cli.exact_states,
        threads: cli.threads.unwrap_or_else(|| CheckConfig::default().threads),
    };

    if let Some(gran) = &cli.por_probe {
        let elem = match gran.as_str() {
            "elem" => true,
            "var" => false,
            other => {
                eprintln!("--por-probe must be 'var' or 'elem', got {other}");
                return ExitCode::from(2);
            }
        };
        let states = inquint::explorer::por_probe(&spec, &cfg, elem);
        println!("[probe:{gran}] {states} states explored (UNSOUND upper-bound probe)");
        return ExitCode::SUCCESS;
    }

    let source = cli.input.display().to_string();
    let write_itf = |trace: &[inquint::state::State], violation: bool| {
        if let Some(path) = &cli.out_itf {
            let itf = inquint::itf_out::trace_to_itf(&spec.vars.names, trace, violation, &source);
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

    let print_trace = |trace: &[inquint::state::State]| {
        for (i, state) in trace.iter().enumerate() {
            eprintln!("State {i}:");
            for (name, value) in spec.vars.names.iter().zip(state.iter()) {
                eprintln!("  {name} = {value}");
            }
        }
    };

    if !spec.temporal.is_empty() {
        use inquint::temporal::{check_temporal, TemporalOutcome};
        if cfg.max_steps.is_some() {
            eprintln!("warning: temporal checking requires exhaustive exploration; ignoring --max-steps");
        }
        let tcfg = CheckConfig {
            max_steps: None,
            ..cfg
        };
        return match check_temporal(&spec, &tcfg) {
            Ok(TemporalOutcome::Pass { states }) => {
                println!(
                    "[ok:assured-exact] {states} states explored exhaustively; \
                     temporal properties hold"
                );
                ExitCode::SUCCESS
            }
            Ok(TemporalOutcome::SafetyViolation { property, trace }) => {
                print_trace(&trace);
                write_itf(&trace, true);
                println!(
                    "[violation] temporal property '{property}' violated at depth {}",
                    trace.len() - 1
                );
                ExitCode::from(1)
            }
            Ok(TemporalOutcome::Violation { property, lasso }) => {
                for (i, state) in lasso.states.iter().enumerate() {
                    if i == lasso.loop_index {
                        eprintln!("──── loop starts here ────");
                    }
                    eprintln!("State {i}:");
                    for (name, value) in spec.vars.names.iter().zip(state.iter()) {
                        eprintln!("  {name} = {value}");
                    }
                }
                if let Some(path) = &cli.out_itf {
                    let mut itf = inquint::itf_out::trace_to_itf(
                        &spec.vars.names,
                        &lasso.states,
                        true,
                        &source,
                    );
                    itf.loop_index = Some(lasso.loop_index as u64);
                    if let Ok(json) = serde_json::to_string_pretty(&itf) {
                        let _ = std::fs::write(path, json);
                    }
                }
                println!(
                    "[violation] temporal property '{property}' violated (lasso: {} prefix + {} loop states)",
                    lasso.loop_index,
                    lasso.states.len() - lasso.loop_index
                );
                ExitCode::from(1)
            }
            Ok(TemporalOutcome::InvariantViolation { invariant, trace }) => {
                print_trace(&trace);
                write_itf(&trace, true);
                println!(
                    "[violation] invariant '{invariant}' violated at depth {}",
                    trace.len() - 1
                );
                ExitCode::from(1)
            }
            Ok(TemporalOutcome::Deadlock { trace }) => {
                print_trace(&trace);
                write_itf(&trace, true);
                println!("[violation] deadlock reached at depth {}", trace.len() - 1);
                ExitCode::from(1)
            }
            Ok(TemporalOutcome::Incomplete { states }) => {
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
        };
    }

    match check(&spec, &cfg) {
        Ok(CheckOutcome::Pass {
            states,
            max_depth,
            scope,
            assurance,
        }) => {
            let assurance = match assurance {
                Assurance::Exact => "assured-exact",
                Assurance::ProbabilisticFingerprint => "non-assured:fingerprint",
            };
            match scope {
                SearchScope::Exhaustive => println!(
                    "[ok:{assurance}] {states} states explored exhaustively; invariants hold"
                ),
                SearchScope::Bounded { max_steps } => println!(
                    "[ok:{assurance}] {states} states explored; no violation through depth \
                     {max_steps} (deepest reached: {max_depth})"
                ),
            }
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
