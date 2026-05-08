# COMPASS docs

Source for everything published at **https://nvlabs.github.io/COMPASS/**.

```
docs/
├── project_page/   # academic landing page (Bulma; served at /)
└── handbook/       # Sphinx + nvidia-sphinx-theme handbook (served at /docs/)
```

## Local preview

From the repo root, build the handbook and serve it together with the
academic landing — same layout as production:

```bash
pip install -r docs/handbook/requirements.txt
make -C docs/handbook html

mkdir -p /tmp/_site
cp -r docs/project_page/. /tmp/_site/
cp -r docs/handbook/_build/html /tmp/_site/docs

python -m http.server -d /tmp/_site 8765
# /         → academic landing
# /docs/    → handbook
```

For live-reload while editing handbook pages (handbook only; no academic
landing):

```bash
sphinx-autobuild docs/handbook docs/handbook/_build/html --port 8000
```

## Deploy

Automatic on every push to `main` that touches `docs/**` or
`.github/workflows/docs.yml`. The workflow at
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) installs
`docs/handbook/requirements.txt`, runs `make html` (which uses
`sphinx-build -W` so any warning fails the build), stages the academic
landing at site root, copies the handbook under `/docs/`, and deploys via
`actions/deploy-pages@v4` — no intermediate `gh-pages` branch.

The legacy `gh_page` branch is a frozen archive only; not built, not deployed.

## Editing tips

- The handbook is **self-contained** — every page owns its content directly.
  No `{include}` transclusions today, so editing a topic means editing one
  file under `docs/handbook/`.
- Each handbook page has an "Edit on GitHub" link in the right-hand sidebar
  that deep-links to the corresponding markdown source. The `edit_uri` in
  [`handbook/conf.py`](handbook/conf.py) makes this work.
- `-W` strict builds catch broken internal links and unresolved
  cross-references. Preview locally before pushing if you're touching nav
  structure or relative links.
- New pages must be referenced from a `{toctree}` directive in
  [`handbook/index.md`](handbook/index.md), otherwise Sphinx warns about
  orphan pages.
