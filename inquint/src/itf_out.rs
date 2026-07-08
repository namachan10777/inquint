//! Counterexample traces in ITF (Informal Trace Format), compatible with
//! quint's `--out-itf` and the ITF trace viewer.
//! See https://apalache-mc.org/docs/adr/015adr-trace.html

use crate::state::State;
use crate::value::{Value, ValueData};
use quint_ast::QuintName;
use std::collections::BTreeMap;

impl Value {
    pub fn to_itf(self) -> itf::Value {
        match self.data() {
            ValueData::Int(i) => itf::Value::Number(*i),
            ValueData::Bool(b) => itf::Value::Bool(*b),
            ValueData::Str(s) => itf::Value::String(s.to_string()),
            ValueData::Set(s) => itf::Value::Set(s.iter().map(|v| v.to_itf()).collect()),
            ValueData::Tuple(vs) => itf::Value::Tuple(vs.iter().map(|v| v.to_itf()).collect()),
            ValueData::List(vs) => itf::Value::List(vs.iter().map(|v| v.to_itf()).collect()),
            ValueData::Record(shape, values) => itf::Value::Record(
                shape
                    .fields
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| (k.to_string(), v.to_itf()))
                    .collect(),
            ),
            ValueData::Map(m) => {
                itf::Value::Map(m.iter().map(|(k, v)| (k.to_itf(), v.to_itf())).collect())
            }
            // Variants are encoded as { tag, value } records, like quint does.
            ValueData::Variant(label, payload) => itf::Value::Record(
                [
                    ("tag".to_string(), itf::Value::String(label.to_string())),
                    ("value".to_string(), payload.to_itf()),
                ]
                .into_iter()
                .collect(),
            ),
            v => panic!("cannot convert to ITF: {v:?} (states are always normalized)"),
        }
    }
}

/// Serialize a trace. States are vectors of per-variable values.
pub fn trace_to_itf(
    var_names: &[QuintName],
    trace: &[State],
    violation: bool,
    source: &str,
) -> itf::Trace<itf::Value> {
    let states = trace
        .iter()
        .zip(0u64..)
        .map(|(state, i)| itf::State {
            meta: itf::state::Meta {
                index: Some(i),
                other: BTreeMap::new(),
            },
            value: itf::Value::Record(
                var_names
                    .iter()
                    .zip(state.iter())
                    .map(|(name, value)| (name.to_string(), value.to_itf()))
                    .collect(),
            ),
        })
        .collect();

    let mut other = BTreeMap::new();
    other.insert(
        "status".to_string(),
        if violation { "violation" } else { "ok" }.to_string(),
    );

    itf::Trace {
        meta: itf::trace::Meta {
            format: Some("ITF".to_string()),
            format_description: Some(
                "https://apalache-mc.org/docs/adr/015adr-trace.html".to_string(),
            ),
            source: Some(source.to_string()),
            description: Some("Created by inquint".to_string()),
            var_types: BTreeMap::default(),
            timestamp: None,
            other,
        },
        vars: var_names.iter().map(|n| n.to_string()).collect(),
        states,
        params: vec![],
        loop_index: None,
    }
}
