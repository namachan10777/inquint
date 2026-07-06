/**
 * Single-decree Paxos, written from scratch.
 *
 * Proposers run phase 1 (prepare) and phase 2 (accept) over a set of
 * acceptors; any two quorums intersect. A value is chosen when a quorum
 * accepts it in the same ballot. Agreement: at most one value is chosen.
 *
 * Unlike the classic TLA+ spec (and the quint port of it, which cannot be
 * verified — https://github.com/informalsystems/quint/issues/1284), the
 * ballot space is bounded (0..MaxBallot), so both backends can enumerate it.
 *
 * Feature showcase: sum types with record payloads + match, fold over a
 * message soup, assume declarations (quorum intersection), tuple
 * destructuring, bounded nondet over consts.
 *
 * Expected-pass:
 *   quint verify specs/Paxos.qnt --main=main \
 *     --invariants agreement oneValuePerBallot --max-steps=8
 *
 * Expected-violation:
 *   quint verify specs/Paxos.qnt --main=main \
 *     --invariant=brokenNothingChosen --max-steps=6      (depth 6)
 */
module paxos {
  const Acceptors: Set[str]
  const Values: Set[int]
  const Quorums: Set[Set[str]]
  const MaxBallot: int

  assume _ = and {
    Quorums.forall(q => q.subseteq(Acceptors)),
    tuples(Quorums, Quorums).forall(((q1, q2)) => q1.intersect(q2) != Set()),
    MaxBallot >= 1,
  }

  val Ballots = 0.to(MaxBallot)

  /// Sentinel for "no ballot / no value yet".
  pure val NoBal = -1
  pure val NoVal = -1

  type Msg =
    | P1a({ bal: int })
    | P1b({ acc: str, bal: int, vBal: int, vVal: int })
    | P2a({ bal: int, v: int })
    | P2b({ acc: str, bal: int, v: int })

  /// Highest ballot promised by each acceptor.
  var maxBal: str -> int
  /// Highest ballot in which each acceptor voted, and the vote's value.
  var maxVBal: str -> int
  var maxVal: str -> int
  /// Message soup: messages are only ever added.
  var msgs: Set[Msg]

  pure def p1bs(ms: Set[Msg]): Set[{ acc: str, bal: int, vBal: int, vVal: int }] =
    ms.fold(Set(), (acc, m) =>
      match m {
        | P1b(r) => acc.union(Set(r))
        | _ => acc
      })

  pure def p2as(ms: Set[Msg]): Set[{ bal: int, v: int }] =
    ms.fold(Set(), (acc, m) =>
      match m {
        | P2a(r) => acc.union(Set(r))
        | _ => acc
      })

  pure def p2bs(ms: Set[Msg]): Set[{ acc: str, bal: int, v: int }] =
    ms.fold(Set(), (acc, m) =>
      match m {
        | P2b(r) => acc.union(Set(r))
        | _ => acc
      })

  action init = all {
    maxBal' = Acceptors.mapBy(_ => NoBal),
    maxVBal' = Acceptors.mapBy(_ => NoBal),
    maxVal' = Acceptors.mapBy(_ => NoVal),
    msgs' = Set(),
  }

  /// Phase 1a: a proposer starts ballot b.
  action phase1a(b: int): bool = all {
    msgs' = msgs.union(Set(P1a({ bal: b }))),
    maxBal' = maxBal,
    maxVBal' = maxVBal,
    maxVal' = maxVal,
  }

  /// Phase 1b: acceptor a promises ballot b and reports its last vote.
  action phase1b(a: str): bool = all {
    nondet b = Ballots.oneOf()
    all {
      msgs.contains(P1a({ bal: b })),
      b > maxBal.get(a),
      maxBal' = maxBal.set(a, b),
      msgs' = msgs.union(Set(P1b({
        acc: a, bal: b, vBal: maxVBal.get(a), vVal: maxVal.get(a),
      }))),
    },
    maxVBal' = maxVBal,
    maxVal' = maxVal,
  }

  /// Phase 2a: with a quorum of 1b messages for ballot b, the proposer
  /// asks acceptors to vote for v: the value of the highest-ballot vote
  /// reported by the quorum, or a free choice if nobody voted yet.
  action phase2a(b: int, v: int): bool = all {
    // at most one 2a per ballot
    p2as(msgs).forall(m => m.bal != b),
    val replies = p1bs(msgs).filter(m => m.bal == b)
    Quorums.exists(q => and {
      q.forall(a => replies.exists(m => m.acc == a)),
      val qReplies = replies.filter(m => q.contains(m.acc))
      val maxVBalInQ = qReplies.fold(NoBal, (mx, m) => if (m.vBal > mx) m.vBal else mx)
      or {
        // nobody in the quorum has voted: any value is safe
        maxVBalInQ == NoBal,
        // otherwise v must be the value voted in the highest ballot
        qReplies.exists(m => m.vBal == maxVBalInQ and m.vVal == v),
      }
    }),
    msgs' = msgs.union(Set(P2a({ bal: b, v: v }))),
    maxBal' = maxBal,
    maxVBal' = maxVBal,
    maxVal' = maxVal,
  }

  /// Phase 2b: acceptor a votes for the 2a proposal of ballot b.
  action phase2b(a: str): bool = all {
    p2as(msgs) != Set(),
    nondet m = p2as(msgs).oneOf()
    all {
      m.bal >= maxBal.get(a),
      maxBal' = maxBal.set(a, m.bal),
      maxVBal' = maxVBal.set(a, m.bal),
      maxVal' = maxVal.set(a, m.v),
      msgs' = msgs.union(Set(P2b({ acc: a, bal: m.bal, v: m.v }))),
    },
  }

  action step =
    nondet b = Ballots.oneOf()
    nondet v = Values.oneOf()
    nondet a = Acceptors.oneOf()
    any {
      phase1a(b),
      phase1b(a),
      phase2a(b, v),
      phase2b(a),
    }

  def chosen(v: int): bool =
    Quorums.exists(q => Ballots.exists(b =>
      q.forall(a => p2bs(msgs).contains({ acc: a, bal: b, v: v }))))

  /// Agreement: at most one value is ever chosen (holds).
  val agreement =
    tuples(Values, Values).forall(((v1, v2)) =>
      (chosen(v1) and chosen(v2)) implies v1 == v2)

  /// A ballot carries at most one proposal (holds).
  val oneValuePerBallot =
    tuples(p2as(msgs), p2as(msgs)).forall(((m1, m2)) =>
      m1.bal == m2.bal implies m1.v == m2.v)

  /// Choosing a value is clearly possible: phase1a, 2 x phase1b,
  /// phase2a, 2 x phase2b — depth 6 (fails).
  val brokenNothingChosen =
    Values.forall(v => not(chosen(v)))
}

module main {
  import paxos(
    Acceptors = Set("a1", "a2", "a3"),
    Values = Set(1, 2),
    Quorums = Set(Set("a1", "a2"), Set("a1", "a3"), Set("a2", "a3")),
    MaxBallot = 2
  ).*
}
