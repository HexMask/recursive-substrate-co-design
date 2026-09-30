# Recursive Substrate Co-Design (RSCD)

**Recursive Substrate Co-Design (RSCD)** is a conceptual research proposal by
**David Holbert, Independent Researcher**. This repository is the public
Version 0.1 release, dated **September 30, 2026**.

> **Status:** conceptual proposal and experimental seed. RSCD has not been
> experimentally validated. The small Generation Zero program here is a toy
> simulation; it is not evidence that RSCD works, nor a simulator of physical
> computing substrates.

## The proposed loop

RSCD asks whether a machine-designed computational substrate could become the
native developmental environment for a successor intelligence:

```text
design substrate → train successor natively → successor designs next substrate → repeat
```

The proposal combines ideas adjacent to automated design, unconventional
computing, learning-rule search, and recursive improvement. Its contribution is
the *generational composition* of those ideas, not a claim that any component
is novel in isolation.

## Contents

- [`paper/`](paper/) — the supplied polished v0.1 PDF, editable DOCX, original ZIP package, and expanded package files.
- [`references/`](references/) — citation metadata and BibTeX.
- [`src/rscd_gen0/`](src/rscd_gen0/) — the intentionally small Generation Zero starter.
- [`tests/`](tests/) — basic functionality tests.

## Generation Zero starter

The starter represents candidate substrates through a deliberately compact set
of parameters:

- activation: `identity`, `tanh`, or `relu`
- update schedule: synchronous or asynchronous
- plasticity and memory persistence
- topology: dense, ring, or local

It runs delayed-recall and signal-propagation toy benchmarks, then performs a
small seeded elitist evolutionary search. Scores are only properties of this
toy implementation; they do not measure intelligence, hardware performance,
or scientific support for RSCD.

### Run

Requires Python 3.10+; the implementation has no runtime dependencies.

```bash
python -m pip install -e .
rscd-gen0 --population 12 --generations 5 --seed 0
```

Or run it without installing:

```bash
PYTHONPATH=src python -m rscd_gen0.cli --population 12 --generations 5 --seed 0
```

### Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Reproducibility and limitations

The command-line seed makes a run repeatable. This repository does not yet
contain substrate-native learning, a baseline comparison, resource accounting,
physical-hardware models, or evidence for inter-generational improvement.
Those omissions are deliberate: the aim is an honest, inspectable starting
point rather than an inflated result.

## Citation

See [`CITATION.cff`](CITATION.cff) and the package's [`RSCD_references.bib`](references/RSCD_references.bib).

## License

Code is released under the MIT License. The paper and other written materials
are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
