//! The name-resolution lookup table: reference-node id -> definition.

use crate::ir::{Declaration, LambdaParam, QuintId, QuintName};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer};

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum LookupDefinition {
    Definition(Declaration),
    Param(LambdaParam),
}

impl LookupDefinition {
    pub fn name(&self) -> Option<&QuintName> {
        match self {
            Self::Definition(Declaration::OpDef(d)) => Some(&d.name),
            Self::Definition(Declaration::Var { name, .. })
            | Self::Definition(Declaration::Assume { name, .. })
            | Self::Definition(Declaration::Const { name, .. }) => Some(name),
            Self::Definition(_) => None,
            Self::Param(p) => Some(&p.name),
        }
    }
}

/// Maps every `Name`/`App` reference id to the definition it refers to.
///
/// The quint compiler serializes this as a JSON object whose keys are the
/// decimal string form of the ids (JSONbig turns bigint map keys into
/// strings), so a custom deserializer parses them back to u64. See
/// serde-rs/json#1254 for why `#[serde(deserialize_with)]` on a
/// `HashMap<u64, _>` is not enough inside tagged enums.
#[derive(Default, Debug)]
pub struct LookupTable(FxHashMap<QuintId, LookupDefinition>);

impl std::ops::Deref for LookupTable {
    type Target = FxHashMap<QuintId, LookupDefinition>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'de> Deserialize<'de> for LookupTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{MapAccess, Visitor};
        use std::fmt;

        struct TableVisitor;

        impl<'de> Visitor<'de> for TableVisitor {
            type Value = LookupTable;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map with string keys holding u64 ids")
            }

            fn visit_map<M>(self, mut map: M) -> Result<LookupTable, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut table = FxHashMap::default();
                while let Some(key) = map.next_key::<String>()? {
                    let id: QuintId = key.parse().map_err(serde::de::Error::custom)?;
                    let value: LookupDefinition = map.next_value()?;
                    table.insert(id, value);
                }
                Ok(LookupTable(table))
            }
        }

        deserializer.deserialize_map(TableVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact string-keyed JSON shape produced by the TS side.
    #[test]
    fn deserializes_string_keys() {
        let json = r#"{
            "4": {"kind":"var","name":"n","typeAnnotation":{"id":1,"kind":"int"},"id":2,"depth":0},
            "6": {"id":6,"kind":"def","name":"init","qualifier":"action",
                  "expr":{"id":5,"kind":"app","opcode":"assign",
                          "args":[{"id":4,"kind":"name","name":"n"},{"id":3,"kind":"int","value":1}]},
                  "depth":0}
        }"#;
        let table: LookupTable = serde_json::from_str(json).unwrap();
        assert!(matches!(
            table.get(&4),
            Some(LookupDefinition::Definition(Declaration::Var { .. }))
        ));
        assert_eq!(table.get(&6).unwrap().name().unwrap().as_ref(), "init");
    }
}
