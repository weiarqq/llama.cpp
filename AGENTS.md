# Instructions for llama.cpp

## 语言 / Language

使用中文回复所有问题。所有输出、解释、摘要均使用中文。

---

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Build

**CMake only** — the root `Makefile` is deprecated and redirects to CMake.

```bash
# CPU build
cmake -B build
cmake --build build --config Release -j 8

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Use presets (Ninja, RPATH configured):
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release

# Build with tests
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Binaries go to `build/bin/`.

## Testing

```bash
# Run all tests
ctest --test-dir build

# Run specific test by name
ctest --test-dir build -R test-tokenizer-0

# Run specific test by regex with debug/build script
./scripts/debug-test.sh test-tokenizer
./scripts/debug-test.sh test-tokenizer 3  # test #3
./scripts/debug-test.sh -g test-tokenizer  # in GDB
```

Run the full local CI before publishing (see [ci/README.md](ci/README.md)):
```bash
mkdir tmp && GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

If you modify `ggml/` source, run `test-backend-ops` to verify backend consistency.

## Python Scripts

```bash
# Lint (flake8)
pip install flake8 flake8-no-print && flake8 .

# Type-check (tyon mypy/pyright via ty.toml)
pip install mypy && mypy .

# Check requirements for convert scripts
./scripts/check-requirements.sh
```

## Pre-commit

```bash
pip install pre-commit
pre-commit run --all-files
```

Checks: trailing whitespace, end-of-file-fixer, YAML, large files, flake8 (Python only).

## Code Style (C/C++)

- 4 spaces, brackets on same line, vertical alignment for readability
- `snake_case` for functions/variables/types
- Enum values: `UPPER_CASE` prefixed with enum name
- Naming pattern: `<class>_<method>`, e.g. `llama_model_init`
- Use `init`/`free` for constructor/destructor actions
- Avoid templates, fancy STL; use basic `for` loops; keep it simple
- Matrix multiplication is unconventional — see [CONTRIBUTING.md](CONTRIBUTING.md) line 96: `C = ggml_mul_mat(ctx, A, B)` means $C^T = A B^T$

## Architecture

- Core library: `src/` + `include/llama.h`
- `ggml/` is a vendored copy of the [ggml library](https://github.com/ggml-org/ggml) — use `scripts/sync-ggml.sh` to update from upstream
- Tools: `tools/`, examples: `examples/`, tests: `tests/`, common utils: `common/`
- Python model conversion scripts: `convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, etc.

## ggml Sync

```bash
./scripts/sync-ggml.sh  # copies files from ../ggml/ into llama.cpp
```

## Key Conventions for PRs

- Search existing issues/PRs first
- New models/features: CPU-only in initial PR; GPU backends in follow-ups
- New quantization types require perplexity, KL divergence, and benchmark comparisons
- Commit format: `<module> : <commit title> (#<issue_number>)` (squash-merge)
- Module list: https://github.com/ggml-org/llama.cpp/wiki/Modules

## Important Docs

- [Build guide](docs/build.md)
- [Server usage](tools/server/README.md)
- [Server development](tools/server/README-dev.md)
- [How to add a model](docs/development/HOWTO-add-model.md)
- [PEG parser](docs/development/parsing.md)
- [Auto parser](docs/autoparser.md)
- [Jinja engine](common/jinja/README.md)
- [Debugging tests](docs/development/debugging-tests.md)
- [PR template](.github/pull_request_template.md)

---

## AI Usage Policy

AI assistance is permissible only when the majority of the code is authored by a human contributor. AI tools may be used for learning, code review suggestions, mechanical formatting tasks, and completing patterns the contributor has already designed.

**Prohibited**: AI-written PR descriptions, AI-generated responses to reviewers, implementing features without understanding the codebase, automated commits.

When AI meaningfully contributes, disclosure is required. Contributors must be able to explain every line of code they submit.

Maintainers will close PRs that violate these standards. This does not apply to private forks.
