/**
 * Lamport's distributed mutual exclusion with logical clocks, written from
 * scratch ("Time, Clocks, and the Ordering of Events", 1978).
 *
 * Each process broadcasts a timestamped request over FIFO channels. A
 * process enters the critical section when its own request has the lowest
 * (timestamp, id) among all requests it knows of, and every other process
 * has acknowledged it. Requests are totally ordered by (ts, id), which
 * yields mutual exclusion.
 *
 * Feature showcase: Maps with tuple keys, List-based FIFO channels,
 * sum types carrying payloads, .reps random-walk run test.
 *
 * Expected-pass:
 *   quint test   specs/LamportMutex.qnt --main=main
 *   quint verify specs/LamportMutex.qnt --main=main \
 *     --invariants mutex requestConsistency --max-steps=8
 *
 * Expected-violation:
 *   quint verify specs/LamportMutex.qnt --main=main \
 *     --invariant=brokenNooneCritical --max-steps=6    (depth 4)
 */
module lamport_mutex {
  const N: int
  /// Bound on request timestamps, keeping the state space finite.
  const MaxClock: int

  assume _ = and {
    N >= 2,
    MaxClock >= 2,
  }

  val Procs = 1.to(N)
  val Pairs = tuples(Procs, Procs).filter(((p, q)) => p != q)

  type Msg = Request(int) | Release | Ack

  type Pc = Idle | Waiting | Critical

  var clock: int -> int
  var pc: int -> Pc
  /// reqTs.get((p, q)): latest request timestamp of q known to p (0 = none).
  var reqTs: (int, int) -> int
  /// acks.get(p): processes that acknowledged p's current request.
  var acks: int -> Set[int]
  /// chan.get((p, q)): FIFO channel of messages in flight from p to q.
  var chan: (int, int) -> List[Msg]

  val allVars = (clock, pc, reqTs, acks, chan)

  pure def isRequest(m: Msg): bool =
    match m {
      | Request(_) => true
      | _ => false
    }

  /// Request (ts1, p1) beats (ts2, p2) in the total order.
  pure def beats(ts1: int, p1: int, ts2: int, p2: int): bool = or {
    ts1 < ts2,
    ts1 == ts2 and p1 < p2,
  }

  action init = all {
    clock' = Procs.mapBy(_ => 0),
    pc' = Procs.mapBy(_ => Idle),
    reqTs' = tuples(Procs, Procs).mapBy(_ => 0),
    acks' = Procs.mapBy(_ => Set()),
    chan' = tuples(Procs, Procs).mapBy(_ => []),
  }

  /// p requests the critical section: bump the clock and broadcast.
  action request(p: int): bool = all {
    pc.get(p) == Idle,
    clock.get(p) < MaxClock,
    val ts = clock.get(p) + 1
    all {
      clock' = clock.set(p, ts),
      reqTs' = reqTs.set((p, p), ts),
      chan' = chan.keys().mapBy(((src, dst)) =>
        if (src == p and dst != p) chan.get((src, dst)).append(Request(ts))
        else chan.get((src, dst))),
    },
    pc' = pc.set(p, Waiting),
    acks' = acks.set(p, Set(p)),
  }

  /// q consumes the head of the channel from p.
  action receive(q: int, p: int): bool = all {
    p != q,
    chan.get((p, q)) != [],
    val m = chan.get((p, q)).head()
    val rest = chan.get((p, q)).tail()
    match m {
      | Request(ts) => all {
          // record the request and send an Ack back
          reqTs' = reqTs.set((q, p), ts),
          chan' = chan
            .set((p, q), rest)
            .setBy((q, p), l => l.append(Ack)),
          clock' = clock.setBy(q, c => if (c > ts) c else ts + 1),
          acks' = acks,
        }
      | Ack => all {
          acks' = acks.setBy(q, a => a.union(Set(p))),
          chan' = chan.set((p, q), rest),
          reqTs' = reqTs,
          clock' = clock,
        }
      | Release => all {
          reqTs' = reqTs.set((q, p), 0),
          chan' = chan.set((p, q), rest),
          acks' = acks,
          clock' = clock,
        }
    },
    pc' = pc,
  }

  /// p enters the critical section: all acks received and p's request is
  /// the minimum outstanding request p knows of.
  action enter(p: int): bool = all {
    pc.get(p) == Waiting,
    pc' = pc.set(p, Critical),
    reqTs.get((p, p)) > 0,
    acks.get(p) == Procs,
    Procs.forall(q =>
      q != p implies or {
        reqTs.get((p, q)) == 0,
        beats(reqTs.get((p, p)), p, reqTs.get((p, q)), q),
      }),
    clock' = clock,
    reqTs' = reqTs,
    acks' = acks,
    chan' = chan,
  }

  /// p leaves the critical section and broadcasts the release.
  action exit(p: int): bool = all {
    pc.get(p) == Critical,
    pc' = pc.set(p, Idle),
    reqTs' = reqTs.set((p, p), 0),
    acks' = acks.set(p, Set()),
    chan' = chan.keys().mapBy(((src, dst)) =>
      if (src == p and dst != p) chan.get((src, dst)).append(Release)
      else chan.get((src, dst))),
    clock' = clock,
  }

  action step =
    nondet p = Procs.oneOf()
    nondet q = Procs.oneOf()
    any {
      request(p),
      receive(q, p),
      enter(p),
      exit(p),
      // resting state: clocks exhausted and everything delivered
      all {
        clock' = clock,
        pc' = pc,
        reqTs' = reqTs,
        acks' = acks,
        chan' = chan,
      },
    }

  /// Mutual exclusion (holds).
  val mutex =
    Pairs.forall(((p, q)) =>
      not(pc.get(p) == Critical and pc.get(q) == Critical))

  /// A process in the critical section still has its own request recorded,
  /// and channels never carry two concurrent requests from the same sender
  /// (holds).
  val requestConsistency = and {
    Procs.forall(p =>
      pc.get(p) == Critical implies reqTs.get((p, p)) > 0),
    Pairs.forall(((p, q)) =>
      chan.get((p, q)).select(isRequest).length() <= 1),
  }

  /// The critical section is clearly reachable (fails at depth 4 with
  /// N = 2: request, receive Request, receive Ack, enter — the requester's
  /// self-ack is recorded at request time).
  val brokenNooneCritical =
    Procs.forall(p => pc.get(p) != Critical)
}

module main {
  import lamport_mutex(N = 2, MaxClock = 3).*

  /// A full enter/exit round trip for process 1.
  run enterExitTest =
    init
      .then(request(1))
      .then(receive(2, 1))    // 2 consumes Request, sends Ack
      .then(receive(1, 2))    // 1 consumes Ack
      .then(enter(1))
      .expect(pc.get(1) == Critical)
      .then(exit(1))
      .then(receive(2, 1))    // 2 consumes Release
      .expect(pc.get(1) == Idle and reqTs.get((2, 1)) == 0)

  /// Random exploration never violates mutual exclusion.
  run randomWalkTest =
    init.then(15.reps(_ => step)).expect(mutex)
}
