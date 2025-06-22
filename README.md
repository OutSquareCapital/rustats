# Rustats

## Compilation

Build with
````
$env:RUSTFLAGS="--cfg pyo3_disable_reference_pool"
maturin build --release
````

## Installation

````
uv add git+https://github.com/OutSquareCapital/rustats.git
````

## Update

````
uv sync --upgrade
````
