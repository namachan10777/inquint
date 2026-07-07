//! SCC analysis of the product: find a strongly connected component that
//! (i) intersects every GBA acceptance set and (ii) admits a fair cycle
//! under the property's WF/SF premises — TLC-style, with the standard
//! strong-fairness refinement recursion.

use super::buchi::Gba;
use super::graph::StateGraph;
use super::parse::Fairness;
use super::product::Product;
use rustc_hash::FxHashMap;

/// A fair accepting SCC: the final (possibly SF-refined) node set, plus
/// the witnesses the counterexample cycle must visit.
pub struct FairScc {
    pub nodes: Vec<u32>,
    /// One product node per acceptance set.
    pub acc_witnesses: Vec<u32>,
    /// Product edges (source node, target node, csr edge) that must be on
    /// the cycle (taken-fairness witnesses).
    pub edge_witnesses: Vec<(u32, u32, u32)>,
    /// Product nodes that must be on the cycle (disabled-fairness
    /// witnesses for WF constraints without a taken edge).
    pub node_witnesses: Vec<u32>,
}

/// Iterative Tarjan SCC over a sub-view of the product.
fn tarjan(adj: &FxHashMap<u32, Vec<(u32, u32)>>, nodes: &[u32]) -> Vec<Vec<u32>> {
    #[derive(Clone, Copy)]
    struct NodeData {
        index: u32,
        lowlink: u32,
        on_stack: bool,
    }

    let mut data: FxHashMap<u32, NodeData> = FxHashMap::default();
    let mut stack: Vec<u32> = Vec::new();
    let mut sccs = Vec::new();
    let mut counter: u32 = 0;

    for &root in nodes {
        if data.contains_key(&root) {
            continue;
        }
        // explicit DFS stack: (node, next child index)
        let mut call_stack: Vec<(u32, usize)> = vec![(root, 0)];
        data.insert(
            root,
            NodeData {
                index: counter,
                lowlink: counter,
                on_stack: true,
            },
        );
        counter += 1;
        stack.push(root);

        while let Some(&mut (v, ref mut child_idx)) = call_stack.last_mut() {
            let children = adj.get(&v).map(|c| c.as_slice()).unwrap_or(&[]);
            if *child_idx < children.len() {
                let (w, _) = children[*child_idx];
                *child_idx += 1;
                match data.get(&w) {
                    None => {
                        data.insert(
                            w,
                            NodeData {
                                index: counter,
                                lowlink: counter,
                                on_stack: true,
                            },
                        );
                        counter += 1;
                        stack.push(w);
                        call_stack.push((w, 0));
                    }
                    Some(wd) if wd.on_stack => {
                        let w_index = wd.index;
                        let vd = data.get_mut(&v).unwrap();
                        vd.lowlink = vd.lowlink.min(w_index);
                    }
                    Some(_) => {}
                }
            } else {
                call_stack.pop();
                let vd = *data.get(&v).unwrap();
                if let Some(&mut (parent, _)) = call_stack.last_mut() {
                    let pd = data.get_mut(&parent).unwrap();
                    pd.lowlink = pd.lowlink.min(vd.lowlink);
                }
                if vd.lowlink == vd.index {
                    let mut scc = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        data.get_mut(&w).unwrap().on_stack = false;
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
            }
        }
    }
    sccs
}

/// Restrict the product adjacency to a node set.
fn induced(
    product: &Product,
    nodes: &[u32],
) -> FxHashMap<u32, Vec<(u32, u32)>> {
    let set: rustc_hash::FxHashSet<u32> = nodes.iter().copied().collect();
    nodes
        .iter()
        .map(|&n| {
            let edges = product.adj[n as usize]
                .iter()
                .filter(|(t, _)| set.contains(t))
                .copied()
                .collect();
            (n, edges)
        })
        .collect()
}

/// Search the whole product for a fair accepting SCC.
pub fn find_fair_accepting(
    product: &Product,
    gba: &Gba,
    graph: &StateGraph,
    fairness: &[Fairness],
) -> Option<FairScc> {
    let all_nodes: Vec<u32> = (0..product.nodes.len() as u32).collect();
    let adj = induced(product, &all_nodes);
    for scc in tarjan(&adj, &all_nodes) {
        if let Some(found) = check_scc(product, gba, graph, fairness, scc) {
            return Some(found);
        }
    }
    None
}

fn check_scc(
    product: &Product,
    gba: &Gba,
    graph: &StateGraph,
    fairness: &[Fairness],
    scc: Vec<u32>,
) -> Option<FairScc> {
    // Trivial SCC (single node without a self-loop) admits no cycle.
    let adj = induced(product, &scc);
    if scc.len() == 1 {
        let n = scc[0];
        if !adj[&n].iter().any(|(t, _)| *t == n) {
            return None;
        }
    }

    // (i) generalized acceptance: every acceptance set must be represented.
    let mut acc_witnesses = Vec::with_capacity(gba.acc_sets.len());
    for acc in &gba.acc_sets {
        match scc.iter().find(|&&p| acc[product.nodes[p as usize].1 as usize]) {
            Some(&w) => acc_witnesses.push(w),
            None => return None,
        }
    }

    // (ii) fairness.
    let taken_edge = |f: &Fairness| -> Option<(u32, u32, u32)> {
        for &p in &scc {
            for &(t, e) in &adj[&p] {
                if graph.edge_bits[f.taken as usize][e as usize] {
                    return Some((p, t, e));
                }
            }
        }
        None
    };
    let disabled_node = |f: &Fairness| -> Option<u32> {
        scc.iter()
            .copied()
            .find(|&p| !graph.state_bits[f.enabled as usize][product.nodes[p as usize].0 as usize])
    };

    let mut edge_witnesses = Vec::new();
    let mut node_witnesses = Vec::new();
    let mut bad_strong: Vec<&Fairness> = Vec::new();

    for f in fairness {
        if let Some(edge) = taken_edge(f) {
            edge_witnesses.push(edge);
            continue;
        }
        if !f.strong {
            // WF without a taken edge: the cycle must visit a state where
            // ⟨A⟩_v is disabled. If none exists, no sub-SCC can help
            // either (both disjuncts are monotone in the node/edge set).
            match disabled_node(f) {
                Some(n) => node_witnesses.push(n),
                None => return None,
            }
        } else {
            // SF without a taken edge: the cycle must avoid every state
            // where ⟨A⟩_v is enabled.
            let has_enabled = scc.iter().any(|&p| {
                graph.state_bits[f.enabled as usize][product.nodes[p as usize].0 as usize]
            });
            if has_enabled {
                bad_strong.push(f);
            }
        }
    }

    if bad_strong.is_empty() {
        return Some(FairScc {
            nodes: scc,
            acc_witnesses,
            edge_witnesses,
            node_witnesses,
        });
    }

    // SF refinement: remove all states enabling any bad SF action and
    // re-decompose; acceptance and WF must be re-established inside.
    let refined: Vec<u32> = scc
        .into_iter()
        .filter(|&p| {
            let s = product.nodes[p as usize].0;
            !bad_strong
                .iter()
                .any(|f| graph.state_bits[f.enabled as usize][s as usize])
        })
        .collect();
    if refined.is_empty() {
        return None;
    }
    let sub_adj = induced(product, &refined);
    for sub in tarjan(&sub_adj, &refined) {
        if let Some(found) = check_scc(product, gba, graph, fairness, sub) {
            return Some(found);
        }
    }
    None
}
