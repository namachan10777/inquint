/**
 * Folklore reliable broadcast over crash faults, written from scratch.
 *
 * Process 1 broadcasts a payload to all processes. It may crash mid-send,
 * reaching only a subset. On first receipt, a process relays the message
 * to everyone and then delivers it. This gives the classic guarantees:
 *   validity  — only the broadcast payload is ever delivered;
 *   totality  — if any correct process delivers, every correct process
 *               eventually delivers (under fairness).
 *
 * Feature showcase: nondet over powerset (partial send), inductive
 * invariant over Map/Set domains, weakFair + leadsTo (TLC backend).
 *
 * Expected-pass:
 *   quint verify specs/ReliableBroadcast.qnt --main=main \
 *     --invariants validity relayedBeforeDelivered
 *   quint verify specs/ReliableBroadcast.qnt --main=main \
 *     --invariant=validity --inductive-invariant=IndInv
 *   quint verify specs/ReliableBroadcast.qnt --main=main --backend=tlc \
 *     --temporal=totality
 *
 * Expected-violation:
 *   quint verify specs/ReliableBroadcast.qnt --main=main \
 *     --invariant=brokenNobodyDelivers --max-steps=3     (depth 2)
 *   quint verify specs/ReliableBroadcast.qnt --main=main --backend=tlc \
 *     --temporal=brokenTotalityNoFairness                (stuttering lasso)
 */
module reliable_broadcast {
  const N: int
  /// Processes that are allowed to crash. Process 1 is the sender.
  const Faulty: Set[int]

  val Procs = 1.to(N)
  val Correct = Procs.exclude(Faulty)

  assume _ = and {
    N >= 2,
    Faulty.subseteq(1.to(N)),
  }

  pure val Sender = 1
  pure val Values = Set("m1", "m2")

  /// The payload chosen by the sender (fixed at init).
  var payload: str
  /// sentTo.get(p): processes to which p has sent/relayed the payload.
  var sentTo: int -> Set[int]
  /// delivered.get(p): payloads p has delivered.
  var delivered: int -> Set[str]
  var crashed: Set[int]

  val allVars = (payload, sentTo, delivered, crashed)

  def hasMsg(p: int): bool = Procs.exists(q => p.in(sentTo.get(q)))

  action init =
    nondet v = Values.oneOf()
    all {
      payload' = v,
      sentTo' = Procs.mapBy(_ => Set()),
      delivered' = Procs.mapBy(_ => Set()),
      crashed' = Set(),
    }

  /// The sender broadcasts. A faulty sender may crash mid-send and reach
  /// only an arbitrary subset of the processes; a correct sender reaches
  /// everyone. (Note: the domain of the nondet must be a plain set —
  /// `(if (..) powerset(..) else ..).oneOf()` is silently unsatisfiable
  /// under Apalache.)
  action bcast = all {
    not(Sender.in(crashed)),
    sentTo.get(Sender) == Set(),
    nondet rcvs = powerset(Procs).oneOf()
    all {
      Sender.in(Faulty) or rcvs == Procs,
      sentTo' = sentTo.set(Sender, rcvs),
      crashed' = if (rcvs == Procs) crashed else crashed.union(Set(Sender)),
    },
    payload' = payload,
    delivered' = delivered,
  }

  /// On first receipt, relay to everyone, then deliver.
  action relay(p: int): bool = all {
    not(p.in(crashed)),
    hasMsg(p),
    sentTo.get(p) == Set(),
    sentTo' = sentTo.set(p, Procs),
    delivered' = delivered.setBy(p, d => d.union(Set(payload))),
    payload' = payload,
    crashed' = crashed,
  }

  action crash(p: int): bool = all {
    p.in(Faulty),
    not(p.in(crashed)),
    crashed' = crashed.union(Set(p)),
    payload' = payload,
    sentTo' = sentTo,
    delivered' = delivered,
  }

  action step =
    nondet p = Procs.oneOf()
    any {
      bcast,
      relay(p),
      crash(p),
      // resting state: everyone done or crashed
      all {
        payload' = payload,
        sentTo' = sentTo,
        delivered' = delivered,
        crashed' = crashed,
      },
    }

  /// Only the broadcast payload is ever delivered (holds).
  val validity =
    Procs.forall(p => delivered.get(p).subseteq(Set(payload)))

  /// A process relays before (atomically with) delivering (holds).
  val relayedBeforeDelivered =
    Procs.forall(p =>
      delivered.get(p) != Set() implies sentTo.get(p) == Procs)

  /// Domain constraint for induction.
  val TypeOK = and {
    payload.in(Values),
    sentTo.in(Procs.setOfMaps(powerset(Procs))),
    delivered.in(Procs.setOfMaps(powerset(Values))),
    crashed.in(powerset(Faulty)),
  }

  /// Inductive invariant proving validity (holds).
  val IndInv = and {
    TypeOK,
    validity,
    relayedBeforeDelivered,
    // only the sender originates; others relay everything or nothing
    Procs.forall(p =>
      p != Sender implies (sentTo.get(p) == Set() or sentTo.get(p) == Procs)),
  }

  /// If some correct process delivers, all correct processes eventually
  /// deliver, given fair relaying (holds).
  temporal totality =
    Correct.forall(p => weakFair(relay(p), allVars))
      implies Correct.forall(p => Correct.forall(q =>
        (delivered.get(p) != Set()) leadsTo (delivered.get(q) != Set())))

  /// Without fairness a correct process may never relay (fails).
  temporal brokenTotalityNoFairness =
    Correct.forall(p => Correct.forall(q =>
      (delivered.get(p) != Set()) leadsTo (delivered.get(q) != Set())))

  /// Delivery is clearly reachable: bcast then relay, depth 2 (fails).
  val brokenNobodyDelivers =
    Procs.forall(p => delivered.get(p) == Set())
}

module main {
  import reliable_broadcast(N = 3, Faulty = Set(1)).*
}
