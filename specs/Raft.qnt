/**
 * Raft leader election and log replication, written from scratch.
 * (No Raft spec exists in the upstream quint examples.)
 *
 * An abstract shared-state model: candidates gather votes directly from
 * peers and leaders push log entries directly to followers, which
 * preserves Raft's safety arguments (election safety, log matching)
 * while keeping the state space small enough for both backends.
 *
 * Two instances:
 *   raft_3        — full model (terms <= 3, logs <= 2): safety invariants
 *   raft_election — election only (MaxLogLen = 0): TLC temporal checks
 *
 * Feature showcase: polymorphic Option[a] (votedFor), records in Lists
 * (logs), quorum arithmetic, orKeep action property, weakFair + leadsTo.
 *
 * Expected-pass:
 *   quint verify specs/Raft.qnt --main=raft_3 \
 *     --invariants electionSafety logMatching voteIntegrity --max-steps=8
 *   quint verify specs/Raft.qnt --main=raft_election --backend=tlc \
 *     --temporal=termsMonotone,quorumCandidateProgress
 *
 * Expected-violation:
 *   quint verify specs/Raft.qnt --main=raft_3 \
 *     --invariant=brokenNoLeader --max-steps=4           (depth 3)
 *   quint verify specs/Raft.qnt --main=raft_3 \
 *     --invariant=brokenAtMostOneCandidate --max-steps=3 (depth 2)
 *   quint verify specs/Raft.qnt --main=raft_election --backend=tlc \
 *     --temporal=brokenEventuallyLeaderNoFairness        (stuttering lasso)
 */
module raft {
  const Servers: Set[str]
  const MaxTerm: int
  const MaxLogLen: int
  const Values: Set[int]

  assume _ = and {
    size(Servers) >= 3,
    MaxTerm >= 1,
    MaxLogLen >= 0,
  }

  type Option[a] = Some(a) | None

  type Role = Follower | Candidate | Leader

  type LogEntry = { term: int, value: int }

  var currentTerm: str -> int
  var role: str -> Role
  var votedFor: str -> Option[str]
  /// votesGranted.get(c): servers that granted their vote to candidate c
  /// in c's current term.
  var votesGranted: str -> Set[str]
  var log: str -> List[LogEntry]

  val allVars = (currentTerm, role, votedFor, votesGranted, log)

  pure def isQuorum(s: Set[str], all_: Set[str]): bool =
    2 * size(s) > size(all_)

  pure def lastTerm(l: List[LogEntry]): int =
    if (l.length() == 0) 0 else l[l.length() - 1].term

  /// Is candidate c's log at least as up-to-date as voter v's? (§5.4.1)
  def upToDate(c: str, v: str): bool = or {
    lastTerm(log.get(c)) > lastTerm(log.get(v)),
    and {
      lastTerm(log.get(c)) == lastTerm(log.get(v)),
      log.get(c).length() >= log.get(v).length(),
    },
  }

  action init = all {
    currentTerm' = Servers.mapBy(_ => 0),
    role' = Servers.mapBy(_ => Follower),
    votedFor' = Servers.mapBy(_ => None),
    votesGranted' = Servers.mapBy(_ => Set()),
    log' = Servers.mapBy(_ => []),
  }

  /// s times out and starts an election for the next term.
  /// A candidate that already holds a quorum does not restart the
  /// election (modeling choice that makes quorumCandidateProgress hold).
  action timeout(s: str): bool = all {
    currentTerm.get(s) < MaxTerm,
    role.get(s) != Leader,
    not(role.get(s) == Candidate and isQuorum(votesGranted.get(s), Servers)),
    currentTerm' = currentTerm.setBy(s, t => t + 1),
    role' = role.set(s, Candidate),
    votedFor' = votedFor.set(s, Some(s)),
    votesGranted' = votesGranted.set(s, Set(s)),
    log' = log,
  }

  /// Voter v grants its vote to candidate c. If v is behind, it first
  /// adopts c's term (as a follower with a fresh vote).
  action grantVote(v: str, c: str): bool = all {
    v != c,
    role.get(c) == Candidate,
    upToDate(c, v),
    // v may only vote in c's term, and only once
    or {
      currentTerm.get(v) < currentTerm.get(c),
      and {
        currentTerm.get(v) == currentTerm.get(c),
        votedFor.get(v).in(Set(None, Some(c))),
        role.get(v) != Leader,
      },
    },
    currentTerm' = currentTerm.set(v, currentTerm.get(c)),
    role' = role.set(v, Follower),
    votedFor' = votedFor.set(v, Some(c)),
    votesGranted' = votesGranted.setBy(c, g => g.union(Set(v))),
    log' = log,
  }

  /// A candidate with a quorum of votes becomes leader.
  action becomeLeader(s: str): bool = all {
    role.get(s) == Candidate,
    isQuorum(votesGranted.get(s), Servers),
    role' = role.set(s, Leader),
    currentTerm' = currentTerm,
    votedFor' = votedFor,
    votesGranted' = votesGranted,
    log' = log,
  }

  /// A leader appends a client request to its own log.
  action clientRequest(s: str): bool = all {
    role.get(s) == Leader,
    log.get(s).length() < MaxLogLen,
    nondet v = Values.oneOf()
    log' = log.setBy(s, l => l.append({ term: currentTerm.get(s), value: v })),
    currentTerm' = currentTerm,
    role' = role,
    votedFor' = votedFor,
    votesGranted' = votesGranted,
  }

  /// Leader l replicates its log prefix to follower f: f adopts l's term
  /// and its log becomes a (longer) prefix of l's log. Copying whole
  /// prefixes models AppendEntries' consistency check abstractly and
  /// preserves the log-matching property by construction.
  action replicate(l: str, f: str): bool = all {
    l != f,
    role.get(l) == Leader,
    currentTerm.get(f) <= currentTerm.get(l),
    // f's log must be a strict prefix of l's log
    log.get(f).length() < log.get(l).length(),
    log.get(f).indices().forall(i => log.get(f)[i] == log.get(l)[i]),
    log' = log.set(f, log.get(l).slice(0, log.get(f).length() + 1)),
    currentTerm' = currentTerm.set(f, currentTerm.get(l)),
    role' = role.set(f, Follower),
    votedFor' = votedFor,
    votesGranted' = votesGranted,
  }

  action step =
    nondet s = Servers.oneOf()
    nondet t = Servers.oneOf()
    any {
      timeout(s),
      grantVote(s, t),
      becomeLeader(s),
      clientRequest(s),
      replicate(s, t),
      // resting state: terms exhausted, logs full, everything replicated
      all {
        currentTerm' = currentTerm,
        role' = role,
        votedFor' = votedFor,
        votesGranted' = votesGranted,
        log' = log,
      },
    }

  /// At most one leader per term (holds).
  val electionSafety =
    tuples(Servers, Servers).forall(((s, t)) =>
      (s != t and role.get(s) == Leader and role.get(t) == Leader)
        implies currentTerm.get(s) != currentTerm.get(t))

  /// If two logs agree on the term at some index, they agree on all
  /// earlier entries too (holds).
  val logMatching =
    tuples(Servers, Servers).forall(((s, t)) =>
      log.get(s).indices().forall(i =>
        (i < log.get(t).length() and log.get(s)[i].term == log.get(t)[i].term)
          implies 0.to(i).forall(j => log.get(s)[j] == log.get(t)[j])))

  /// Votes counted by a candidate are backed by real votedFor entries,
  /// unless the voter has already moved to a newer term (holds).
  val voteIntegrity =
    Servers.forall(c =>
      role.get(c) == Candidate implies
        votesGranted.get(c).forall(v => or {
          votedFor.get(v) == Some(c),
          currentTerm.get(v) > currentTerm.get(c),
        }))

  /// Every transition adds a non-negative delta to each server's term.
  action termsKeepOrGrow =
    nondet deltas = Servers.setOfMaps(0.to(MaxTerm)).oneOf()
    currentTerm' = Servers.mapBy(s => currentTerm.get(s) + deltas.get(s))

  /// Terms never decrease — action property [][termsKeepOrGrow]_allVars (holds).
  temporal termsMonotone = always(orKeep(termsKeepOrGrow, allVars))

  /// Quorum test for temporal properties: quint 0.32's TLA emission for
  /// TLC is broken by size() applied to state inside a temporal formula,
  /// so majority is spelled out as "two distinct voters" (valid for the
  /// 3-server instances used below).
  def hasTwoVotes(s: str): bool =
    tuples(Servers, Servers).exists(((a, b)) => and {
      a != b,
      a.in(votesGranted.get(s)),
      b.in(votesGranted.get(s)),
    })

  /// A candidate holding a quorum eventually stops being a candidate
  /// (it becomes leader), under fairness of becomeLeader (holds).
  temporal quorumCandidateProgress =
    Servers.forall(s => weakFair(becomeLeader(s), allVars))
      implies Servers.forall(s =>
        (role.get(s) == Candidate and hasTwoVotes(s))
          leadsTo (role.get(s) != Candidate))

  /// Without fairness nobody is obliged to act: the initial stuttering
  /// lasso already violates this (fails).
  temporal brokenEventuallyLeaderNoFairness =
    eventually(Servers.exists(s => role.get(s) == Leader))

  /// A leader is clearly reachable: timeout, grantVote, becomeLeader —
  /// depth 3 (fails).
  val brokenNoLeader = Servers.forall(s => role.get(s) != Leader)

  /// Concurrent candidates are legal: two timeouts — depth 2 (fails).
  val brokenAtMostOneCandidate =
    size(Servers.filter(s => role.get(s) == Candidate)) <= 1
}

module raft_3 {
  import raft(
    Servers = Set("s1", "s2", "s3"),
    MaxTerm = 3,
    MaxLogLen = 2,
    Values = Set(1, 2)
  ).*
}

module raft_election {
  import raft(
    Servers = Set("s1", "s2", "s3"),
    MaxTerm = 2,
    MaxLogLen = 0,
    Values = Set(1)
  ).*
}
