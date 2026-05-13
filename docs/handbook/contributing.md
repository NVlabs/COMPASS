# Contributing

The contribution rules — sign-off, commit policy, pre-commit, code style —
live in
[`CONTRIBUTING.md`](https://github.com/NVlabs/COMPASS/blob/main/CONTRIBUTING.md):

```{include} ../../CONTRIBUTING.md
```

## Editing this handbook

Each page in this site has an "Edit on GitHub" button (top right) that
deep-links to the corresponding markdown source under
`docs/handbook/`. Edit, push, and the GitHub Actions workflow at
[`.github/workflows/docs.yml`](https://github.com/NVlabs/COMPASS/blob/main/.github/workflows/docs.yml)
will re-build and re-deploy automatically on push to `main`.

To preview locally:

```bash
pip install -r docs/handbook/requirements.txt
cd docs/handbook && make html
# Built to _build/html/ — open with any static server:
python -m http.server -d _build/html 8000

# Or use sphinx-autobuild for live reload while editing:
sphinx-autobuild . _build/html --port 8000
```

The handbook builds with `-W` (warnings → errors), so any broken internal
link or unresolved reference fails CI.
