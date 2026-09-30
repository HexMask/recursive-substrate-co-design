import unittest

from rscd_gen0 import Candidate, evolve, run_benchmarks


class GenerationZeroTests(unittest.TestCase):
    def test_candidate_step_keeps_a_persistent_memory_trace(self):
        candidate = Candidate(
            activation="identity",
            update_mode="synchronous",
            plasticity=0.0,
            memory_persistence=0.8,
            topology="ring",
            size=4,
        )

        first = candidate.step([1.0, 0.0, 0.0, 0.0])
        second = candidate.step([0.0, 0.0, 0.0, 0.0])

        self.assertGreater(first[0], 0)
        self.assertGreater(second[0], 0)


    def test_toy_benchmarks_return_bounded_named_scores(self):
        result = run_benchmarks(Candidate.default(), seed=4)

        self.assertEqual(set(result.scores), {"memory", "signal"})
        self.assertTrue(all(0.0 <= score <= 1.0 for score in result.scores.values()))
        self.assertTrue(0.0 <= result.fitness <= 1.0)


    def test_evolution_produces_scored_candidates_for_each_generation(self):
        history = evolve(population_size=6, generations=3, seed=7)

        self.assertEqual(len(history), 3)
        self.assertTrue(all(len(generation) == 6 for generation in history))
        self.assertTrue(
            all(0.0 <= scored.result.fitness <= 1.0 for generation in history for scored in generation)
        )
