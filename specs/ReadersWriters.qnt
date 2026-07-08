/**
 * Readers-writers with a FIFO request queue, written from scratch.
 *
 * Processes enqueue read or write requests. Requests are served strictly
 * in queue order: a read is served when no writer is active, a write when
 * nobody is active. FIFO order plus weak fairness of serving gives
 * starvation freedom — a property the classic "readers preference"
 * solution does not have.
 *
 * Feature showcase: sum types in a List-based FIFO queue, higher-order
 * List operators (select/head/tail), weakFair + leadsTo (TLC backend).
 *
 * Expected-pass:
 *   quint verify specs/ReadersWriters.qnt --main=main --invariant=safety
 *   quint verify specs/ReadersWriters.qnt --main=main --backend=tlc \
 *     --temporal=noStarvation
 *
 * Expected-violation:
 *   quint verify specs/ReadersWriters.qnt --main=main \
 *     --invariant=brokenOneReader --max-steps=5          (depth 4)
 *   quint verify specs/ReadersWriters.qnt --main=main --backend=tlc \
 *     --temporal=brokenNoStarvationNoFairness            (stuttering lasso)
 */
module readers_writers {
  const N: int

  assume _ = N >= 2

  val Procs = 1.to(N)

  type Req = ReadReq(int) | WriteReq(int)

  var readers: Set[int]
  var writers: Set[int]
  var queue: List[Req]

  val allVars = (readers, writers, queue)

  pure def requester(r: Req): int =
    match r {
      | ReadReq(p) => p
      | WriteReq(p) => p
    }

  def isActive(p: int): bool = p.in(readers) or p.in(writers)
  def isQueued(p: int): bool = queue.select(r => requester(r) == p) != []

  action init = all {
    readers' = Set(),
    writers' = Set(),
    queue' = [],
  }

  action requestRead(p: int): bool = all {
    not(isActive(p)),
    not(isQueued(p)),
    queue' = queue.append(ReadReq(p)),
    readers' = readers,
    writers' = writers,
  }

  action requestWrite(p: int): bool = all {
    not(isActive(p)),
    not(isQueued(p)),
    queue' = queue.append(WriteReq(p)),
    readers' = readers,
    writers' = writers,
  }

  /// Serve the request at the head of the queue, if permitted.
  action serveHead = all {
    queue != [],
    writers == Set(),
    match queue.head() {
      | ReadReq(p) => all {
          readers' = readers.union(Set(p)),
          writers' = writers,
        }
      | WriteReq(p) => all {
          readers == Set(),
          readers' = readers,
          writers' = writers.union(Set(p)),
        }
    },
    queue' = queue.tail(),
  }

  /// An active process finishes its read or write.
  action release(p: int): bool = all {
    isActive(p),
    readers' = readers.exclude(Set(p)),
    writers' = writers.exclude(Set(p)),
    queue' = queue,
  }

  action step =
    nondet p = Procs.oneOf()
    any {
      requestRead(p),
      requestWrite(p),
      serveHead,
      release(p),
      // resting state: nobody queued or active
      all { readers' = readers, writers' = writers, queue' = queue },
    }

  /// Safety: at most one writer, and never readers and writers together (holds).
  val safety = and {
    size(writers) <= 1,
    writers != Set() implies readers == Set(),
    readers.intersect(writers) == Set(),
  }

  /// FIFO + fair serving and releasing = no starvation (holds).
  temporal noStarvation =
    and {
      weakFair(serveHead, allVars),
      Procs.forall(p => weakFair(release(p), allVars)),
    } implies Procs.forall(p => isQueued(p) leadsTo isActive(p))

  /// Without fairness a queued request may be ignored forever (fails).
  temporal brokenNoStarvationNoFairness =
    Procs.forall(p => isQueued(p) leadsTo isActive(p))

  /// Concurrent readers are allowed, so this is violated at depth 4:
  /// requestRead(1), requestRead(2), serveHead, serveHead (fails).
  val brokenOneReader = size(readers) <= 1
}

module main {
  import readers_writers(N = 3).*
}

// Larger instance for benchmarking explicit-state checkers.
module bench {
  import readers_writers(N = 8).*
}
