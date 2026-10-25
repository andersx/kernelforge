# conda-forge recipes

Copy-pasteable **v1** (`recipe.yaml`) recipes for submitting KernelForge to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes).

| Directory | Package | Notes |
|-----------|---------|--------|
| `kernelforge/` | CPU `kernelforge` | Submit this first |
| `kernelforge-cuda/` | CUDA companion | Submit after CPU is on conda-forge (or in the same staged-recipes PR) |

These recipes are **not** built by this repo’s GitHub Actions. conda-forge CI
builds them after the staged-recipes PR merges and creates the feedstock(s).

## Submit CPU `kernelforge`

1. Fork and clone [staged-recipes](https://github.com/conda-forge/staged-recipes).
2. From this repo:

   ```bash
   mkdir -p /path/to/staged-recipes/recipes/kernelforge
   cp conda/kernelforge/recipe.yaml conda/kernelforge/build.sh \
     /path/to/staged-recipes/recipes/kernelforge/
   ```

3. Open a PR against `conda-forge/staged-recipes` with only that recipe folder.
4. After merge, conda-forge creates `kernelforge-feedstock` and publishes to the
   `conda-forge` channel.

Optional local smoke (requires [`rattler-build`](https://rattler-build.prefix.dev/)):

```bash
rattler-build build --recipe conda/kernelforge
```

## Submit `kernelforge-cuda`

There is no separate PyPI sdist for the companion; the recipe builds from the
**`kernelforge` sdist** (`packaging/kernelforge-cuda/`).

1. Prefer waiting until `kernelforge` is installable from `conda-forge`.
2. Copy `conda/kernelforge-cuda/{recipe.yaml,build.sh,conda_build_config.yaml}`
   into `staged-recipes/recipes/kernelforge-cuda/`.
3. Open a staged-recipes PR (or add the folder alongside CPU in one PR).
4. Reviewers may adjust `conda_build_config.yaml` CUDA versions to match the
   current conda-forge pytorch/CUDA matrix.

The CUDA recipe:

- skips non-Linux and `cuda_compiler_version == "None"`
- builds with `KF_WITH_CUDA=ON` / `KF_CUDA_ONLY=ON` against conda-forge `pytorch`
- sets `TORCH_CUDA_ARCH_LIST` from `CF_TORCH_CUDA_ARCH_LIST`
- depends on `kernelforge` at the same version at runtime

## Version bumps

When releasing on PyPI:

1. Bump `context.version` and `source.sha256` in both recipes.
2. Reset `build.number` to `0` for a new upstream version.
3. After feedstocks exist, prefer the feedstock repos (autotick bot often opens
   the version-bump PR).

```bash
curl -sL "https://pypi.org/pypi/kernelforge/${VERSION}/json" \
  | python -c 'import json,sys; u=[x for x in json.load(sys.stdin)["urls"] if x["packagetype"]=="sdist"][0]; print(u["digests"]["sha256"])'
```
