/**
 * A two-layered write-back cache, written from scratch.
 *
 * Writes land in L1 and mark the key dirty; a background flusher copies
 * dirty entries to L2; clean L1 entries may be evicted. The write-back
 * discipline guarantees that evicting a clean entry never loses data:
 * every clean L1 entry is already backed by an identical L2 entry.
 *
 * Feature showcase: type aliases, action properties via orKeep/next
 * ("box-bracket" [][ver' >= ver]_vars style), mustChange, always/eventually,
 * weakFair + leadsTo (TLC backend).
 *
 * Expected-pass:
 *   quint verify specs/TwoLayeredCache.qnt --main=main \
 *     --invariants cleanConsistency dirtyInL1
 *   quint verify specs/TwoLayeredCache.qnt --main=main --backend=tlc \
 *     --temporal=verMonotone,eventuallyClean
 *   echo y | quint verify specs/TwoLayeredCache.qnt --main=main \
 *     --temporal=verNeverDecreases
 *   (next-based action property: TLC only accepts []-ed actions in [A]_v
 *    form, so this one is checked with Apalache's bounded temporal mode;
 *    `echo y` confirms the partial-support prompt)
 *
 * Expected-violation:
 *   quint verify specs/TwoLayeredCache.qnt --main=main \
 *     --invariant=brokenL1Backed --max-steps=2           (depth 1)
 *   echo y | quint verify specs/TwoLayeredCache.qnt --main=main \
 *     --temporal=brokenAlwaysProgress                    (stuttering lasso;
 *     Apalache backend: quint 0.32 emits malformed TLA for mustChange,
 *     so the TLC backend cannot parse it)
 *   quint verify specs/TwoLayeredCache.qnt --main=main --backend=tlc \
 *     --temporal=brokenEventuallyCleanNoFairness         (lazy flusher lasso)
 */
module two_layered_cache {
  /// Bound on the number of writes, keeping the state space finite for TLC.
  const MaxVer: int

  assume _ = MaxVer >= 1

  type CacheLayer = str -> int

  pure val Keys = Set("k1", "k2")

  var l1: CacheLayer
  var l2: CacheLayer
  /// Keys whose L1 value has not yet been flushed to L2.
  var dirty: Set[str]
  /// Version counter; doubles as the value written.
  var ver: int

  val allVars = (l1, l2, dirty, ver)

  action init = all {
    l1' = Map(),
    l2' = Map(),
    dirty' = Set(),
    ver' = 0,
  }

  /// A client writes the next version to some key in L1 (write-back).
  action write = all {
    ver < MaxVer,
    nondet k = Keys.oneOf()
    all {
      l1' = l1.put(k, ver + 1),
      dirty' = dirty.union(Set(k)),
    },
    ver' = ver + 1,
    l2' = l2,
  }

  /// The background flusher copies one dirty entry down to L2.
  action flushOne = all {
    dirty != Set(),
    nondet k = dirty.oneOf()
    all {
      l2' = l2.put(k, l1.get(k)),
      dirty' = dirty.exclude(Set(k)),
    },
    l1' = l1,
    ver' = ver,
  }

  /// A clean entry may be evicted from L1 at any time.
  action evict = all {
    l1.keys().exclude(dirty) != Set(),
    nondet k = l1.keys().exclude(dirty).oneOf()
    l1' = l1.keys().exclude(Set(k)).mapBy(kk => l1.get(kk)),
    l2' = l2,
    dirty' = dirty,
    ver' = ver,
  }

  action step = any {
    write,
    flushOne,
    evict,
    // resting state once MaxVer writes have happened and all is flushed
    all { l1' = l1, l2' = l2, dirty' = dirty, ver' = ver },
  }

  /// Write-back correctness: clean L1 entries are backed by L2 (holds).
  val cleanConsistency =
    Keys.forall(k =>
      (l1.keys().contains(k) and not(dirty.contains(k)))
        implies (l2.keys().contains(k) and l1.get(k) == l2.get(k)))

  /// Dirty keys always live in L1 — they cannot be evicted (holds).
  val dirtyInL1 = dirty.subseteq(l1.keys())

  /// The version counter either stays or increments by exactly one.
  action verAdvance = any { ver' = ver, ver' = ver + 1 }

  /// Action property [][verAdvance]_allVars: versions never go back and
  /// never skip, expressed with the stuttering operator orKeep (holds).
  temporal verMonotone = always(orKeep(verAdvance, allVars))

  /// The same monotonicity, phrased with next (holds).
  temporal verNeverDecreases = always(next(ver) >= ver)

  /// A fair flusher eventually drains the dirty set (holds: writes are
  /// bounded by MaxVer, so flushing cannot be outrun forever).
  temporal eventuallyClean =
    weakFair(flushOne, allVars)
      implies (dirty != Set() leadsTo dirty == Set())

  /// <<step>>_allVars cannot happen infinitely often: the system may (and
  /// eventually must) stutter forever, so this fails on any lasso that
  /// stops making real progress (fails).
  temporal brokenAlwaysProgress =
    always(eventually(mustChange(step, allVars)))

  /// Without fairness the flusher may never run (fails).
  temporal brokenEventuallyCleanNoFairness =
    dirty != Set() leadsTo dirty == Set()

  /// Write-through would guarantee this; write-back does not.
  /// Counterexample at depth 1: the first write is dirty in L1 only (fails).
  val brokenL1Backed =
    Keys.forall(k => l1.keys().contains(k) implies l2.keys().contains(k))
}

module main {
  import two_layered_cache(MaxVer = 3).*
}

// Larger instance for benchmarking explicit-state checkers.
module bench {
  import two_layered_cache(MaxVer = 120).*
}
