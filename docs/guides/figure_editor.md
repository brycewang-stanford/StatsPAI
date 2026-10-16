# Editing figures: `sp.interactive`

`sp.interactive(fig)` edits the look of a matplotlib figure, in the spirit of
Stata's Graph Editor. Titles, labels, fonts, colours, the legend, layout and
annotations can change. The statistical content cannot: points, fitted lines,
error bars, confidence bands and bars are locked.

Every edit is recorded as a named operation. That gives three guarantees,
each tested pixel for pixel on every operation and on random sequences of
operations (`tests/test_figure_editor.py`):

1. `editor.generate_code()`, run after the code that drew the figure,
   reproduces the edited figure exactly.
2. `editor.undo()` and `editor.reset()` return the figure to exactly what it
   was.
3. A saved edit list (`editor.to_edits_json()`) replayed on a freshly drawn
   copy of the figure gives the same result.

The editor never changes global matplotlib settings. `sp.set_theme` is the
function for that, and it affects figures drawn afterwards.

## Three ways to use it

**In Jupyter** (`pip install ipywidgets`), the panel shows a live preview on
the left and five tabs on the right (Theme, Text, Style, Layout, Export).
Undo, redo, reset and switching between subplots keep every control in step
with the figure. When an edit is adjusted or refused to keep the data visible,
the status line says why.

```python
import statspai as sp

mp = sp.datasets.mpdta()
gt = sp.callaway_santanna(data=mp, y="lemp", t="year", i="countyreal", g="first_treat")
fig, ax = sp.ggdid(sp.aggte(gt, type="dynamic", bstrap=False))

editor = sp.interactive(fig)
```

**In a script with a GUI backend** (Qt, Tk, macOS), a small control window
opens next to the figure. Click a title, axis label or annotation, type the
new text, press Enter. The buttons are Undo, Redo, Reset, Grid, Spines, Theme
and Code. The figure itself gets no buttons and keeps its margins.

**Without a user interface** (scripts on the Agg backend, servers, agents),
`sp.interactive` returns the editor and leaves the figure alone. Edits go
through methods or a list of actions:

```python
editor = sp.interactive(fig, actions=[
    {"action": "set_title", "text": "Effect of minimum wage on teen employment"},
    {"action": "apply_theme", "theme": "aea"},
    {"action": "apply_palette", "palette": "okabe_ito"},
    {"action": "set_figsize", "width": 6.5, "height": 4.0},
])
editor.save("figure1.pdf")
print(editor.generate_code())
```

`editor.operations()` lists every operation with its typed parameters, and
`editor.get_editable_schema()` describes the figure (axes, lines, colour
groups, texts, legend) for building a panel or for an agent to plan edits.
Unknown operations or parameters raise `ValueError`.

## What is locked

Each artist gets a role when the editor opens.

| Role | Examples | What can change |
| --- | --- | --- |
| `DATA` | points, data lines, bars, colour-mapped scatter, images | colour, width, style, marker (but not invisible) |
| `FIT` | lines labelled fit, fitted, trend, predicted | as `DATA` |
| `CI` | error bars and caps, `fill_between` bands | colour, alpha down to 0.05 |
| `REFERENCE` | `axhline`, `axvline`, two-point flat segments, `axvspan` shading | anything |
| `LABEL`, `LEGEND`, `SPINE`, `ANNOTATION` | titles, legend, spines, free text | anything |

The rules behind the table:

- Errorbar, bar and stem plots are recognised from matplotlib's containers, so
  an event study's point estimates are data and its caps are intervals.
- Only geometry makes a line a reference line. A data series labelled
  "Treatment group" stays locked. A flat line counts as a reference line when
  it is drawn in axes coordinates (`axhline`), has two points, or is labelled
  as a reference (zero, baseline, cutoff, threshold, onset).
- Whatever the editor does not recognise is locked.

With `protect_data=True` (the default):

- Axis limits that would hide a point, an error bar, a band or a bar are
  widened to cover them plus a 2% margin, with a warning. Log and inverted
  axes are handled.
- Data cannot be made invisible. Alpha stops at 0.2 for data and fits and at
  0.05 for intervals. A line style of `"None"` on a line without markers, or
  removing the markers of a marker-only series, raises `ValueError`.
- Colours of colour-mapped artists encode values and cannot be changed.
- Text drawn by the plotting function that contains a number (`"0.124***"`,
  `"N = 500"`) reports a result. Its wording is locked; it can be restyled or
  hidden.

A refused edit leaves the figure exactly as it was.

## Series colours and palettes

A point estimate, its error bars and its confidence band are normally drawn in
one colour. `editor.color_groups(ax_index)` groups the locked artists by that
colour, and recolouring a group changes all of them together:

```python
editor.color_groups(0)          # [{"color": "#1f77b4", "members": [("line", 0), ("coll", 2), ...]}, ...]
editor.set_series_color(0, "#D55E00")
editor.apply_palette("okabe_ito")   # one colour per group, on every axes
```

Legend entries follow: matplotlib's legend keeps copies of the artists made
when it was drawn, so after every recolour or restyle the editor rebuilds the
entries with `sp.refresh_legend(ax)` (also usable on its own), and the
generated code calls it too.

Palettes: the StatsPAI theme palettes (`academic`, `aea`, `minimal`,
`cn_journal`), `okabe_ito` (distinguishable with the common forms of colour
blindness), `tab10` and `grayscale`, or `colors=[...]`. `apply_theme` uses
the theme's palette the same way unless `recolor=False`.

## Saved styles

An edit list is a style that can be applied to other figures, like a Stata
`.grec` recording. Edits that do not fit the new figure (a line it does not
have) are skipped with a warning.

```python
style = editor.to_edits_json()
sp.interactive(other_fig, ui="none").apply_edits_json(style)
```

The JSON holds operations and parameters only. A `code` field in an incoming
record is never executed or copied into generated code. Files written by
earlier versions (records with `target` / `property` / `new`) are translated
to operations.

## Export for a journal

```python
editor.save("figure1.pdf")                      # TrueType fonts embedded
editor.export("figure1", formats=("pdf", "png", "svg"), dpi=600)
editor.check()                                  # small text, grayscale clashes, low contrast
png = editor.preview(grayscale=True)            # how it prints in black and white
```

`save` embeds TrueType (Type 42) fonts in PDF and PostScript output.
Matplotlib's default Type 3 fonts are rejected by many submission systems.
`check()` reports text smaller than 6 pt at the current figure size, series
that cannot be told apart in grayscale (similar luminance and the same line
style and marker), and series drawn almost in the background colour.

Set the figure size to the printed size (`set_figsize(6.5, 4)` for a full
AER page width, `3.5` inches for one journal column) before choosing font
sizes; sizes are in points at that size.

## The generated code

```python
ax = fig.axes[0]
ax.title.set_text('Effect of minimum wage on teen employment')
fig.set_facecolor('#ffffff')
...
```

It needs only the name `fig` and defines `ax`, `ax1`, ... itself. An edit is
left out when a later edit sets the same property, so the code is short and
runs in the order that gives the final result. It contains no
`fig.tight_layout()` unless that edit was made. Theme edits are written out
property by property, because `sp.set_theme` would not change a figure that
is already drawn.

## Troubleshooting

**The panel shows as text (`HBox(children=...)`) instead of controls.** The
notebook front end has not loaded the ipywidgets extension. Install
`ipywidgets` in the kernel's environment and `jupyterlab_widgets` in the
environment that runs JupyterLab, then restart JupyterLab. If both are
installed and the text still appears, JupyterLab is not looking in the
directory that holds the extension: `jupyter labextension list` shows where
it looks, and `JUPYTER_PATH=<prefix>/share/jupyter jupyter lab` adds a
directory (on a Homebrew install the prefix is `/opt/homebrew`).

**A warning says a font is not installed.** Matplotlib draws the text in the
family's fallback font. The edit and its code still work; install the font
to see it.

**An axis limit came out wider than requested.** With `protect_data=True`
limits are widened to keep every point, error bar and band visible; the
warning gives the data range. Pass `protect_data=False` only for figures
without statistical content.

## Changes from earlier versions

See `MIGRATION.md` (figure editor). In short: the generated code changed
form and now reproduces the figure; themes and fonts apply to the figure
instead of global settings; `sp.interactive` no longer adds buttons to
figures on non-interactive backends; `save` embeds TrueType fonts.
