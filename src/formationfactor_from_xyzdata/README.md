# Formation factors from XYZ data

Run the commands below from the repository root.

Chloride, electrical logs and REGIS II
-------------------------------------

Run `python src\formationfactor_from_xyzdata\build_dataset.py` from the repository root.
Use `--help` to override the input paths, match tolerances, or output directory. The
script uses NumPy, pandas, SciPy, xarray and a NetCDF backend (`netCDF4`).
It writes `chloride_geophysics_regis.csv` and `summary.json` to
`data\3-input\formationfactor_from_xyzdata` by default.

- Each chloride observation (g/L) gets at most one ECPT and one LAS match.
  Within each source, an in-screen sample within 250 m in RD (EPSG:28992)
  beats any outside-screen sample. Inside the screen, prefer the closest XY,
  then the closest elevation to the chloride sample. If none are inside,
  prefer the smallest gap to the screen (up to 1 m), then use the same
  tie-breakers. Final ties use source record order.
- Chloride screen bounds and LAS `z` are treated as metres NAP; ECPT elevation
  is surface elevation minus logged depth. The dataset retains source IDs,
  CSV record numbers (header counted as row 1), XY separation, vertical gaps,
  conductivity (S/m) and its reciprocal, resistivity (ohm m). Nonpositive
  conductivity, negative logged depths and invalid required inputs are
  excluded from the logs. Positive values are not independently validated.
- REGIS II's local 100 m grid is sampled independently at the chloride and
  log coordinates using the explicit cell bounds. The dataset retains the
  layer codes, top/bottom elevations, match status and fraction of the
  chloride screen inside its assigned layer. It keeps missing or ambiguous
  assignments and preserves the chloride input's existing REGIS label
  separately. A same-layer match does not imply exact co-location.
- GeoST 0.5.1 does not parse raw LAS well logs or expose a REGIS reader.
  This workflow uses the cleaned LAS CSV and reads the local REGIS II NetCDF
  with xarray. For raw LAS inspection, use `lasio` and verify curve mnemonics,
  units and depth reference before interpreting values as ground conductivity.

To plot conductivity against chloride and reproduce the regression statistics, run
`python src\formationfactor_from_xyzdata\plot_chloride_conductivity.py` (requires
Matplotlib). Outputs go to `reports\figures\formationfactor_from_xyzdata`.
The script fits ordinary least squares to the pooled pairs and to each source separately.
It reports pointwise 95% confidence bands for the mean fit from 5,000 source-stratified
bootstrap draws of whole logs, and leave-one-log-out errors for individual observations.
The regression table also includes a sensitivity fit restricted to in-screen matches
within 25 m with the same REGIS layer at both locations. The confidence bands do not
account for systematic measurement, matching or lithology effects.

Interactive REGIS explorer
--------------------------

The additional chloride figure plots bulk EC25 directly against chloride (g/L)
using exactly the same filtered pairs, shared selections and exclusions. Both
before/after models and confidence bands are transformed onto the chloride axis;
the fixed-minimum constraint remains defined at zero groundwater EC, not zero
chloride. This figure has independent linear axes, scroll zoom and reset, and
clicking a point toggles its shared selection.

Run `python src\formationfactor_from_xyzdata\build_explorer.py`, then open
`reports\figures\formationfactor_from_xyzdata\explorer.html` in a browser.
This builds a single self-contained HTML file using Python's standard library:
no server, package installation, internet connection or third-party data service
is needed. Rebuild after changing the matched input CSV; `--input` and `--output`
can override the default paths.

- Filter by formation group, exact REGIS layer, electrical source, XY distance,
  screen overlap and matching layer at both locations. Switch between REGIS
  assignments at the chloride observation and electrical log. Formation names
  are grouped from the input descriptions before their first comma, using exact
  input REGIS codes; unavailable labels remain explicitly unmapped.
- The interactive app plots calculated groundwater EC on X and bulk EC on Y,
  both in mS/cm at 25 degrees C: `ECbulk,25 = 1.36 * 10 * conductivity_s_m` and
  `ECgroundwater,25 = (1000 * chloride_g_l + 450) / 360`.
  This inverts `Cl[mg/L] = 360 * EC25[mS/cm] - 450` and applies
  `EC25 = 1.36 * EC11`. Measured bulk conductivity is assumed to represent
  11 degrees C and multiplied by 1.36 to obtain EC25. Groundwater EC is
  positive for nonnegative chloride, so no clipping is applied. Downloads
  preserve original measurements and calculated EC at 25 and 11 degrees C.
- Scroll or drag to zoom, double-click to reset, and optionally use logarithmic
  axes. Zoom and axis scaling do not change the original-unit OLS regression.
  Nonpositive values are hidden on logarithmic axes but retained in the fit.
  Regression is refitted as bulk EC against groundwater EC, not inverted from the
  earlier chloride regression. Conversion-factor uncertainty is not included.
- Click points, select a rectangle in selection mode, or check table rows, then
  exclude the selection. Alternatively test a bulk-EC ceiling (mS/cm) or a one-pass
  residual-MAD rule. Compare the before/after lines, slope, intercept, R-squared,
  residual RMSE and whole-log bootstrap uncertainty (1,500 fixed-seed draws).
  Confidence bands describe the mean relation, not individual predictions.
- Manual exclusions persist across filters for the current page session.
  Restore them separately or reset the whole experiment. Download all filtered
  records with exclusion flags and a regression comparison; inputs are never
  modified. Reloading the page resets the experiment.
- Sparse or constant-chloride subsets show an explicit explanation instead of
  a fitted line or unsupported confidence intervals. Data-driven exclusions
  make subsequent confidence intervals exploratory, not confirmatory.
- The app reports formation factor and surface conductivity at 25 degrees C.
  With `ECbulk = a + b * ECgroundwater`, the model
  `ECbulk = ECgroundwater / F + sigma_surface` gives `F = 1/b` and
  `sigma_surface = a` in mS/cm (divide by 10 for S/m).
  An additional least-squares fit fixes `a = min(ECbulk,25)` separately for
  each before/after subset, with `b = sum(x*(y-a))/sum(x*x)`. Both fits
  are plotted and included in the before/after tables and regression export.
  The fixed-minimum bootstrap holds this observed minimum constant; intervals
  are conditional on that constraint and do not estimate uncertainty in the
  minimum. Its intercept interval is therefore degenerate by construction.
  Formation-factor intervals transform each bootstrap slope as `1/b`; they
  are suppressed if the slope interval reaches zero or below. Zero slopes
  have no finite formation factor. Negative surface
  conductivity and F below 1 are explicitly flagged, never forced into a
  physical range. Both parameters, intervals and warnings are included in
  the before/after comparison and regression CSV export.
  Lines extend to x = 0 on linear axes to expose the intercept (extrapolation).
  Centred R-squared can be negative for the fixed-intercept fit.
