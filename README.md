# Risk-Annotated Scene Graphs for Industrial Pick-and-Place

This repository contains the LaTeX source and supporting materials for a master's thesis on risk-annotated scene graphs for industrial robotic pick-and-place.

## Scope

The work investigates how a scene graph of an industrial work cell can be built from RGB-D camera observations and augmented with evidence-backed risk information. It focuses on object states, labels, object parts, and spatial or semantic relations that together determine manipulation hazards.

## Repository layout

- `main.tex` — thesis entry point.
- `Sections/` — thesis chapters.
- `Frontmatter/` — abstracts, declaration, nomenclature, and other preliminary pages.
- `Images/` and `Frontpage/` — figures and title-page assets.
- `references.bib` — bibliography database.
- `Code/` — supporting example code.

## Building the thesis

Use a LaTeX distribution with `latexmk`, `biber`, `minted`, and Pygments available. From the repository root:

```sh
latexmk -pdf -shell-escape main.tex
```

The generated PDF is `main.pdf`. Build artifacts and minted caches are excluded through `.gitignore`.

## Branches

- `main` contains the repository landing page and current thesis source.
- `Documentation` contains the original documentation branch pushed from the local project.
