//! LTL formula IR over opaque atoms, negation-normal form, and the
//! safety-fragment classifier.

use std::fmt;

pub type StateAtomId = u32;
pub type EdgeAtomId = u32;

/// LTL over state atoms (evaluated per state) and edge atoms (evaluated
/// per transition (s, t), including the implicit stutter edge (s, s)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ltl {
    True,
    False,
    SAtom(StateAtomId),
    EAtom(EdgeAtomId),
    Not(Box<Ltl>),
    And(Vec<Ltl>),
    Or(Vec<Ltl>),
    Always(Box<Ltl>),
    Eventually(Box<Ltl>),
}

impl fmt::Display for Ltl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ltl::True => write!(f, "true"),
            Ltl::False => write!(f, "false"),
            Ltl::SAtom(i) => write!(f, "s{i}"),
            Ltl::EAtom(i) => write!(f, "e{i}"),
            Ltl::Not(p) => write!(f, "!({p})"),
            Ltl::And(ps) => {
                write!(f, "(")?;
                for (i, p) in ps.iter().enumerate() {
                    if i > 0 {
                        write!(f, " & ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ")")
            }
            Ltl::Or(ps) => {
                write!(f, "(")?;
                for (i, p) in ps.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ")")
            }
            Ltl::Always(p) => write!(f, "G({p})"),
            Ltl::Eventually(p) => write!(f, "F({p})"),
        }
    }
}

/// A signed atom reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lit {
    pub atom: AtomRef,
    pub pos: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomRef {
    S(StateAtomId),
    E(EdgeAtomId),
}

impl Lit {
    pub fn negated(self) -> Lit {
        Lit {
            atom: self.atom,
            pos: !self.pos,
        }
    }
}

/// Negation normal form with Until/Release, input to the GPVW tableau.
/// `F p = true U p`, `G p = false R p`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Nnf {
    True,
    False,
    Lit(Lit),
    And(Box<Nnf>, Box<Nnf>),
    Or(Box<Nnf>, Box<Nnf>),
    Until(Box<Nnf>, Box<Nnf>),
    Release(Box<Nnf>, Box<Nnf>),
}

impl fmt::Display for Nnf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nnf::True => write!(f, "true"),
            Nnf::False => write!(f, "false"),
            Nnf::Lit(Lit { atom, pos }) => {
                if !pos {
                    write!(f, "!")?;
                }
                match atom {
                    AtomRef::S(i) => write!(f, "s{i}"),
                    AtomRef::E(i) => write!(f, "e{i}"),
                }
            }
            Nnf::And(a, b) => write!(f, "({a} & {b})"),
            Nnf::Or(a, b) => write!(f, "({a} | {b})"),
            Nnf::Until(a, b) => write!(f, "({a} U {b})"),
            Nnf::Release(a, b) => write!(f, "({a} R {b})"),
        }
    }
}

/// Convert to NNF, pushing negations to the atoms.
pub fn nnf(formula: &Ltl, negate: bool) -> Nnf {
    match formula {
        Ltl::True => {
            if negate {
                Nnf::False
            } else {
                Nnf::True
            }
        }
        Ltl::False => {
            if negate {
                Nnf::True
            } else {
                Nnf::False
            }
        }
        Ltl::SAtom(i) => Nnf::Lit(Lit {
            atom: AtomRef::S(*i),
            pos: !negate,
        }),
        Ltl::EAtom(i) => Nnf::Lit(Lit {
            atom: AtomRef::E(*i),
            pos: !negate,
        }),
        Ltl::Not(p) => nnf(p, !negate),
        Ltl::And(ps) => fold_binary(ps, negate, |a, b| {
            if negate {
                Nnf::Or(Box::new(a), Box::new(b))
            } else {
                Nnf::And(Box::new(a), Box::new(b))
            }
        }),
        Ltl::Or(ps) => fold_binary(ps, negate, |a, b| {
            if negate {
                Nnf::And(Box::new(a), Box::new(b))
            } else {
                Nnf::Or(Box::new(a), Box::new(b))
            }
        }),
        // G p = false R p; ¬G p = true U ¬p
        Ltl::Always(p) => {
            let inner = nnf(p, negate);
            if negate {
                Nnf::Until(Box::new(Nnf::True), Box::new(inner))
            } else {
                Nnf::Release(Box::new(Nnf::False), Box::new(inner))
            }
        }
        // F p = true U p; ¬F p = false R ¬p
        Ltl::Eventually(p) => {
            let inner = nnf(p, negate);
            if negate {
                Nnf::Release(Box::new(Nnf::False), Box::new(inner))
            } else {
                Nnf::Until(Box::new(Nnf::True), Box::new(inner))
            }
        }
    }
}

fn fold_binary(ps: &[Ltl], negate: bool, join: impl Fn(Nnf, Nnf) -> Nnf) -> Nnf {
    let mut iter = ps.iter().map(|p| nnf(p, negate));
    let first = iter.next().unwrap_or(if negate { Nnf::False } else { Nnf::True });
    iter.fold(first, join)
}

/// A propositional (temporal-operator-free) formula over atoms.
#[derive(Debug, Clone)]
pub enum Prop {
    True,
    False,
    SAtom(StateAtomId),
    EAtom(EdgeAtomId),
    Not(Box<Prop>),
    And(Vec<Prop>),
    Or(Vec<Prop>),
}

impl Prop {
    /// Does this proposition mention any edge atom? (State-only props can
    /// be checked per state; edge props need the transition.)
    pub fn has_edge_atoms(&self) -> bool {
        match self {
            Prop::True | Prop::False | Prop::SAtom(_) => false,
            Prop::EAtom(_) => true,
            Prop::Not(p) => p.has_edge_atoms(),
            Prop::And(ps) | Prop::Or(ps) => ps.iter().any(|p| p.has_edge_atoms()),
        }
    }
}

/// One conjunct of a safety-fragment property.
pub enum SafetyConjunct {
    /// Checked on initial states only.
    Initial(Prop),
    /// `always(prop)`: checked on every state (state part) and every edge
    /// including the stutter self-loop (edge part).
    AlwaysProp(Prop),
}

/// Classify `body` into the safety fragment: a top-level conjunction where
/// each conjunct is either propositional or `always(propositional)`.
/// Returns None if any part needs the general (Büchi) path.
pub fn safety_fragment(body: &Ltl) -> Option<Vec<SafetyConjunct>> {
    let mut out = Vec::new();
    let conjuncts: Vec<&Ltl> = match body {
        Ltl::And(ps) => ps.iter().collect(),
        other => vec![other],
    };
    for c in conjuncts {
        match c {
            Ltl::Always(p) => out.push(SafetyConjunct::AlwaysProp(as_prop(p)?)),
            other => out.push(SafetyConjunct::Initial(as_prop(other)?)),
        }
    }
    Some(out)
}

fn as_prop(f: &Ltl) -> Option<Prop> {
    Some(match f {
        Ltl::True => Prop::True,
        Ltl::False => Prop::False,
        Ltl::SAtom(i) => Prop::SAtom(*i),
        Ltl::EAtom(i) => Prop::EAtom(*i),
        Ltl::Not(p) => Prop::Not(Box::new(as_prop(p)?)),
        Ltl::And(ps) => Prop::And(ps.iter().map(as_prop).collect::<Option<Vec<_>>>()?),
        Ltl::Or(ps) => Prop::Or(ps.iter().map(as_prop).collect::<Option<Vec<_>>>()?),
        Ltl::Always(_) | Ltl::Eventually(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(i: u32) -> Ltl {
        Ltl::SAtom(i)
    }

    #[test]
    fn nnf_negates_leads_to() {
        // ¬G(p → F q) = true U (p & (false R ¬q))
        let leads_to = Ltl::Always(Box::new(Ltl::Or(vec![
            Ltl::Not(Box::new(s(0))),
            Ltl::Eventually(Box::new(s(1))),
        ])));
        let neg = nnf(&leads_to, true);
        assert_eq!(neg.to_string(), "(true U (s0 & (false R !s1)))");
    }

    #[test]
    fn safety_fragment_accepts_always_prop() {
        let f = Ltl::And(vec![
            Ltl::Always(Box::new(Ltl::EAtom(0))),
            Ltl::Always(Box::new(Ltl::SAtom(1))),
        ]);
        assert!(safety_fragment(&f).is_some());
        let g = Ltl::Always(Box::new(Ltl::Eventually(Box::new(s(0)))));
        assert!(safety_fragment(&g).is_none());
    }
}
