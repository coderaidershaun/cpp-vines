# cpp-vines

## Install

Install Boost and uv on macOS:

```shell
brew install boost uv
```

If you would like to use `python`:

```shell
brew install uv
uv venv
source .venv/bin/activate
uv sync
```

## Run a C++ example

```shell
cmake -S . -B build
cmake --build build --target example-vine
./build/example-vine
```

## Run a Python example

You may call the C++ library directly from Python (this is supported by the `py-bridge.cpp` layer).

See `python/examples/vine.ipynb` for example.
