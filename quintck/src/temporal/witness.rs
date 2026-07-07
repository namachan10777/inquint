//! Lasso counterexample extraction: a prefix from an initial product node
//! to the fair accepting SCC, plus a cycle inside the SCC visiting every
//! acceptance and fairness witness.

use super::graph::StateGraph;
use super::product::Product;
use super::scc::FairScc;
use super::Lasso;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Shortest path (start..=goal node list) via BFS, restricted to `allowed`
/// when given. Returns `[start]` if a start satisfies the goal.
fn bfs_path(
    product: &Product,
    allowed: Option<&FxHashSet<u32>>,
    starts: &[u32],
    goal: impl Fn(u32) -> bool,
) -> Option<Vec<u32>> {
    let ok = |n: u32| allowed.is_none_or(|a| a.contains(&n));
    let mut parent: FxHashMap<u32, u32> = FxHashMap::default();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    let mut queue: VecDeque<u32> = VecDeque::new();

    let build = |end: u32, parent: &FxHashMap<u32, u32>| {
        let mut path = vec![end];
        while let Some(&p) = parent.get(path.last().unwrap()) {
            path.push(p);
        }
        path.reverse();
        path
    };

    for &s in starts {
        if ok(s) && seen.insert(s) {
            if goal(s) {
                return Some(vec![s]);
            }
            queue.push_back(s);
        }
    }
    while let Some(v) = queue.pop_front() {
        for &(w, _) in &product.adj[v as usize] {
            if ok(w) && seen.insert(w) {
                parent.insert(w, v);
                if goal(w) {
                    return Some(build(w, &parent));
                }
                queue.push_back(w);
            }
        }
    }
    None
}

pub fn extract_lasso(product: &Product, graph: &StateGraph, scc: &FairScc) -> Lasso {
    let in_scc: FxHashSet<u32> = scc.nodes.iter().copied().collect();
    let root = scc.nodes[0];

    // Build the cycle root -> ... -> root, visiting all witnesses.
    let mut cycle: Vec<u32> = vec![root];
    let mut current = root;

    let goto = |target: u32, cycle: &mut Vec<u32>, current: &mut u32| {
        let path = bfs_path(product, Some(&in_scc), &[*current], |n| n == target)
            .expect("witness reachable within SCC");
        cycle.extend(&path[1..]);
        *current = target;
    };

    for &w in scc.acc_witnesses.iter().chain(&scc.node_witnesses) {
        goto(w, &mut cycle, &mut current);
    }
    for &(src, dst, _) in &scc.edge_witnesses {
        goto(src, &mut cycle, &mut current);
        cycle.push(dst);
        current = dst;
    }

    // Close the cycle with at least one real edge.
    if cycle.len() == 1 {
        // No witness moved us: force one step to an in-SCC successor.
        let (succ, _) = *product.adj[current as usize]
            .iter()
            .find(|(t, _)| in_scc.contains(t))
            .expect("nontrivial SCC has an internal edge");
        cycle.push(succ);
        current = succ;
    }
    if current != root {
        let path = bfs_path(product, Some(&in_scc), &[current], |n| n == root)
            .expect("SCC is strongly connected");
        cycle.extend(&path[1..]);
    }
    // cycle = root, ..., root (length >= 2)

    // Prefix over the FULL product from an initial node to root.
    let prefix =
        bfs_path(product, None, &product.init, |n| n == root).expect("SCC reachable from init");

    // Lasso node list: prefix (ends at root) + cycle interior; the loop
    // closes from the last node back to root = states[loop_index].
    let mut nodes: Vec<u32> = prefix;
    let loop_index = nodes.len() - 1;
    nodes.extend(&cycle[1..cycle.len() - 1]);

    let states = nodes
        .iter()
        .map(|&p| graph.states[product.nodes[p as usize].0 as usize].clone())
        .collect();

    Lasso { states, loop_index }
}
