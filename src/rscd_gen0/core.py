from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class Candidate:
    """A configurable, simulated computational substrate.

    This is not a neural model or a physical hardware simulator.  Its small
    parameter space makes it useful only as a transparent starting point for
    toy experiments.
    """

    activation: str
    update_mode: str
    plasticity: float
    memory_persistence: float
    topology: str
    size: int = 8
    state: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.activation not in {"identity", "tanh", "relu"}:
            raise ValueError("activation must be identity, tanh, or relu")
        if self.update_mode not in {"synchronous", "asynchronous"}:
            raise ValueError("update_mode must be synchronous or asynchronous")
        if self.topology not in {"dense", "ring", "local"}:
            raise ValueError("topology must be dense, ring, or local")
        if self.size < 2:
            raise ValueError("size must be at least 2")
        self.plasticity = _clamp(self.plasticity)
        self.memory_persistence = _clamp(self.memory_persistence)
        if not self.state:
            self.state = [0.0] * self.size
        if len(self.state) != self.size:
            raise ValueError("state length must match size")

    @classmethod
    def default(cls) -> "Candidate":
        return cls("tanh", "synchronous", 0.1, 0.5, "ring")

    def fresh(self) -> "Candidate":
        return replace(self, state=[0.0] * self.size)

    def _neighbors(self, index: int) -> list[int]:
        if self.topology == "dense":
            return list(range(self.size))
        if self.topology == "ring":
            return [(index - 1) % self.size, index, (index + 1) % self.size]
        return [max(0, index - 1), index]

    def _activate(self, value: float) -> float:
        if self.activation == "identity":
            return value
        if self.activation == "relu":
            return max(0.0, value)
        return math.tanh(value)

    def step(self, inputs: list[float]) -> list[float]:
        """Advance the simulated substrate by one update."""
        if len(inputs) != self.size:
            raise ValueError("inputs length must match size")

        previous = self.state[:]
        next_state = self.state[:] if self.update_mode == "asynchronous" else [0.0] * self.size
        for index in range(self.size):
            source = next_state if self.update_mode == "asynchronous" else previous
            neighbor_average = sum(source[j] for j in self._neighbors(index)) / len(self._neighbors(index))
            retained = self.memory_persistence * previous[index]
            mixed = (1.0 - self.memory_persistence) * (inputs[index] + neighbor_average)
            updated = self._activate(retained + mixed)
            next_state[index] = (1.0 - self.plasticity) * updated + self.plasticity * neighbor_average
        self.state = next_state
        return self.state[:]


@dataclass(frozen=True)
class BenchmarkResult:
    scores: dict[str, float]

    @property
    def fitness(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


def run_benchmarks(candidate: Candidate, seed: int = 0) -> BenchmarkResult:
    """Run two deterministic toy tasks: delayed recall and signal propagation."""
    rng = random.Random(seed)
    memory_machine = candidate.fresh()
    target = rng.randrange(memory_machine.size)
    pulse = [0.0] * memory_machine.size
    pulse[target] = 1.0
    memory_machine.step(pulse)
    for _ in range(3):
        memory_machine.step([0.0] * memory_machine.size)
    memory_score = _clamp((memory_machine.state[target] + 1.0) / 2.0)

    signal_machine = candidate.fresh()
    signal_machine.step([1.0] + [0.0] * (signal_machine.size - 1))
    for _ in range(signal_machine.size):
        signal_machine.step([0.0] * signal_machine.size)
    signal_score = _clamp(sum(abs(value) for value in signal_machine.state) / signal_machine.size)
    return BenchmarkResult({"memory": memory_score, "signal": signal_score})


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    result: BenchmarkResult


def _random_candidate(rng: random.Random) -> Candidate:
    return Candidate(
        activation=rng.choice(["identity", "tanh", "relu"]),
        update_mode=rng.choice(["synchronous", "asynchronous"]),
        plasticity=rng.random(),
        memory_persistence=rng.random(),
        topology=rng.choice(["dense", "ring", "local"]),
        size=rng.choice([4, 6, 8]),
    )


def _mutate(parent: Candidate, rng: random.Random) -> Candidate:
    changes = {
        "plasticity": _clamp(parent.plasticity + rng.uniform(-0.2, 0.2)),
        "memory_persistence": _clamp(parent.memory_persistence + rng.uniform(-0.2, 0.2)),
    }
    if rng.random() < 0.25:
        changes["activation"] = rng.choice(["identity", "tanh", "relu"])
    if rng.random() < 0.25:
        changes["topology"] = rng.choice(["dense", "ring", "local"])
    return replace(parent, **changes, state=[])


def evolve(population_size: int = 12, generations: int = 5, seed: int = 0) -> list[list[ScoredCandidate]]:
    """Run an intentionally simple elitist evolutionary search."""
    if population_size < 2 or generations < 1:
        raise ValueError("population_size must be at least 2 and generations at least 1")
    rng = random.Random(seed)
    population = [_random_candidate(rng) for _ in range(population_size)]
    history: list[list[ScoredCandidate]] = []
    for generation_index in range(generations):
        scored = [ScoredCandidate(item, run_benchmarks(item, seed + generation_index)) for item in population]
        scored.sort(key=lambda item: item.result.fitness, reverse=True)
        history.append(scored)
        parents = [item.candidate for item in scored[: max(1, population_size // 2)]]
        population = [parents[0]] + [_mutate(rng.choice(parents), rng) for _ in range(population_size - 1)]
    return history
