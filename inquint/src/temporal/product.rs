//! Product of the stutter-closed state graph with a GBA.

use super::buchi::Gba;
use super::graph::StateGraph;
use super::ltl::{AtomRef, Lit};
use rustc_hash::FxHashMap;

pub struct Product {
    /// (state id, gba node)
    pub nodes: Vec<(u32, u32)>,
    /// adjacency: (target product node, underlying CSR edge index)
    pub adj: Vec<Vec<(u32, u32)>>,
    pub init: Vec<u32>,
}

pub fn sat_state(g: &StateGraph, s: u32, lits: &[Lit]) -> bool {
    lits.iter().all(|lit| match lit.atom {
        AtomRef::S(i) => g.state_bits[i as usize][s as usize] == lit.pos,
        AtomRef::E(_) => true,
    })
}

pub fn sat_edge(g: &StateGraph, e: u32, lits: &[Lit]) -> bool {
    lits.iter().all(|lit| match lit.atom {
        AtomRef::E(i) => g.edge_bits[i as usize][e as usize] == lit.pos,
        AtomRef::S(_) => true,
    })
}

pub fn build_product(g: &StateGraph, gba: &Gba) -> Product {
    let mut nodes = Vec::new();
    let mut index: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut adj: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut init = Vec::new();
    let mut worklist = Vec::new();

    let mut intern = |s: u32,
                      q: u32,
                      nodes: &mut Vec<(u32, u32)>,
                      adj: &mut Vec<Vec<(u32, u32)>>,
                      worklist: &mut Vec<u32>|
     -> u32 {
        if let Some(&id) = index.get(&(s, q)) {
            return id;
        }
        let id = nodes.len() as u32;
        nodes.push((s, q));
        adj.push(Vec::new());
        index.insert((s, q), id);
        worklist.push(id);
        id
    };

    for s in 0..g.init_count {
        for &q in &gba.init {
            if sat_state(g, s, &gba.state_lits[q as usize]) {
                let id = intern(s, q, &mut nodes, &mut adj, &mut worklist);
                init.push(id);
            }
        }
    }

    while let Some(p) = worklist.pop() {
        let (s, q) = nodes[p as usize];
        for (e, t) in g.edges_of(s) {
            if !sat_edge(g, e, &gba.edge_lits[q as usize]) {
                continue;
            }
            for &q2 in &gba.edges[q as usize] {
                if sat_state(g, t, &gba.state_lits[q2 as usize]) {
                    let tid = intern(t, q2, &mut nodes, &mut adj, &mut worklist);
                    adj[p as usize].push((tid, e));
                }
            }
        }
    }

    Product { nodes, adj, init }
}
