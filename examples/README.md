# Examples

This directory contains regular user-facing RoboKots examples.

Performance benchmarks and developer comparison tools were moved to
`developer/benchmarks`. See `developer/README.md` for those commands.

## Simple Example

```bash
uv run python -m examples.simple_example.main
```

Runs a small end-to-end example with the sample robot model and target list.

## Polars Example

```bash
uv run python -m examples.polars_example.main
```

Exports state data to JSONL, reads it with Polars, and plots the result with
Matplotlib. Install the optional table and visualization dependencies first if
needed:

```bash
uv sync --extra table --extra viz
```

## Models

Sample robot model JSON files live in `examples/model`.
