/**
 * Tiny state machines with *known* liveness counterexample shapes, used
 * as deterministic detection tests for temporal model checkers.
 *
 * - stutter_lab: `eventually(x == 3)` fails via a pure stuttering lasso
 *   (the counter may stop at any point; TLA behaviors always allow
 *   infinite stuttering).
 * - cycle_lab: a modulo-3 counter; `eventually(x == 5)` fails via a real
 *   3-state cycle (5 is unreachable), while `always(eventually(x == 0))`
 *   holds even without fairness on the cycle instance because every
 *   non-stuttering run revisits 0 — but the stuttering lasso breaks it,
 *   so it fails without fairness and holds with weakFair(step).
 * - sf_lab: an action enabled only on alternate states — weak fairness
 *   does not force it (it is never *continuously* enabled), strong
 *   fairness does. Discriminates WF from SF handling.
 */
module stutter_lab {
  var x: int

  action init = x' = 0

  action step = any {
    all { x < 3, x' = x + 1 },
    x' = x,
  }

  /// Fails: stutter forever at x < 3 (fails).
  temporal brokenReach = eventually(x == 3)

  /// Holds: with weak fairness on the increment, x reaches 3.
  temporal fairReach =
    weakFair(all { x < 3, x' = x + 1 }, x) implies eventually(x == 3)
}

module cycle_lab {
  var x: int

  action init = x' = 0

  action step = x' = (x + 1) % 3

  /// Fails: 5 is simply unreachable — the counterexample is a real cycle
  /// 0 -> 1 -> 2 -> 0, not a stutter loop (step has no stutter branch,
  /// but TLA stuttering closure also admits stutter lassos) (fails).
  temporal brokenReachFive = eventually(x == 5)

  /// Holds under fairness: every fair run cycles through 0 forever.
  temporal fairRevisitZero =
    weakFair(step, x) implies always(eventually(x == 0))

  /// Fails without fairness: stutter at x == 1 forever (fails).
  temporal brokenRevisitZero = always(eventually(x == 0))
}

module sf_lab {
  var x: int

  action init = x' = 0

  /// Enabled only on even x: intermittently enabled along the main loop.
  action fire = all { x % 2 == 0, x' = 5 }
  action tick = all { x != 5, x' = (x + 1) % 4 }

  action step = any {
    fire,
    tick,
    // terminal resting state after firing
    all { x == 5, x' = x },
  }

  /// Weak fairness is NOT enough: the fair run 0,1,2,3,0,... enables
  /// `fire` only intermittently, so WF(fire) is vacuously satisfied —
  /// the counterexample is the real 4-state cycle (fails).
  temporal brokenWeakFire =
    (weakFair(tick, x) and weakFair(fire, x)) implies eventually(x == 5)

  /// Strong fairness forces `fire` on the 0,1,2,3 cycle (holds).
  temporal strongFire =
    (weakFair(tick, x) and strongFair(fire, x)) implies eventually(x == 5)
}
