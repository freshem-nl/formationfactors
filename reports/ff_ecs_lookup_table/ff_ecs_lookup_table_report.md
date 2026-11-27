# Formation factor and surface conductivity look-up tables for the FRESHEM Monte Carlo

**Technical report: how `look_up_table_ff_ECs_{geotop,regis}_v4_ln_SIP3_SIP5.csv` were made, plus the side avenues**

Repository: `freshem-nl/formationfactors`, `main` at `aca2b06` (in sync with origin on 2026-11-27).
Final tables: `Nextcloud/FreshEM (Projectfolder)/Team4-Chloride/FormatieFactor/ff_ecs_uncertainty/for_monte_carlo/`, written 2026-08-26.

---

## 1. Summary

The two tables give every combination of stratigraphic unit (GeoTOP or REGIS) and GeoTOP lithoclass a probability distribution for the intrinsic formation factor *F* and the surface conductivity *ECs*. The Monte Carlo chloride workflow (`chloride_uncertainties/src/3-model/hem2chloride.py`) draws from these distributions when it converts HEM resistivity to pore-water EC and chloride.

How the tables were made:

1. **Data.** 357 disturbed lab samples (`FF_Disturbed`) from the 2026 sampling campaign. SIP3 (3-salinity) results are used, and SIP5 (5-salinity) results fill the gaps for 72 samples that have no SIP3 result. Anthropogenic samples (AAOM) are excluded.
2. **Transform.** *F* and *ECs* are both close to lognormal (checked with Shapiro–Wilk tests and Q–Q plots), so all statistics are computed on ln *F* and ln *ECs*.
3. **Grouping.** Kruskal–Wallis tests followed by Dunn post-hoc tests (Benjamini–Hochberg correction, α = 0.1, minimum group size 5) decide whether a lithoclass should be split by depositional facies or by stratigraphy. Facies groups that do not differ significantly are merged by hand.
4. **Assembly.** Each (unit, lithoclass) cell gets the most specific statistic available, in this order:
   1. stratigraphy-specific statistic
   2. facies-within-lithoclass statistic
   3. lithoclass statistic
   4. literature value (gravel, shells) or proxy (anthropogenic → medium sand).
5. **Output.** 828 GeoTOP rows (92 units × 9 lithoclasses) and 1179 REGIS rows (131 units × 9). Mean and standard deviation are given in ln-space with `dist_type = lognorm`. Gravel and shells keep fixed literature values.

I re-derived the lithoclass statistics and all group sizes independently from the lab table, and they match the published SIP3+SIP5 tables exactly (§6).

Side avenues covered in §5:

- Monte Carlo uncertainty propagation for **shells and gravel** via Archie's law.
- An apparent formation factor for **Boom Clay (Boomse klei)**, inferred from HEM resistivity.
- A **FRESHEM-Zeeland** version of the table.
- A **copula** analysis of the dependence between *F* and *ECs*.
- Statistics on **1/F**.
- Checks of the **Revil 2017 and 2026** lab results.
- The **direct ECbulk → chloride** avenue.

**None of the side-avenue results are in the v4 tables yet.**

Section 7 lists four issues that matter for the Monte Carlo results. The most urgent:

- The anthropogenic rows (`a`) are labelled `norm` but contain ln-space values, so they are sampled wrongly. This is fixed in the script, but the published CSV files still need to be regenerated.
- Gravel and shells still have fixed values with zero spread.
- Boom Clay gets generic marine clay statistics (*F* ≈ 3.4), while the HEM-based estimate is *F* ≈ 9.

---

## 2. Input data

| Item | Source |
|---|---|
| Lab results (F, ECs, lithoclass, stratigraphy) | `data/3-input/lab_results/20260304_tbl20_WPchloride_FFdata.xlsx` (origin: `Team1-Fieldwork/05_Sampling&FFresults`) |
| Lab results incl. cementation exponent *m*, grain size | `data/3-input/lab_results/20260126_tbl05_Measurementdata_Full.xlsx` |
| GeoTOP unit → facies table | `data/1-external/GeoTOP_formaties_afzettingsmilieus _JanG_final.csv` (columns `STR_UNIT_CD`, `VOXEL_NR`, `facies`) |
| REGIS unit → facies table | `data/1-external/regis_formaties_afzettingsmilieus _JanG_final.csv` (columns `formation`, `user_nr`, `facies`) |

Lab columns used:

- `SIP3_FormationFactor_F_3W_unitless`, `SIP3_SurfCond_Sigmas_3W_S/m`
- for SIP5 gap filling: `SIP5_FormationFactor_F5_unitless`, `SIP5_SurfCond_Sigmas_S/m`, flag `FF_SIP5 == "Yes"`

**Sample selection.** All `stat_analysis_*` scripts filter the same way:

1. Keep `Type_name == "FF_Disturbed"`: 384 of 479 rows. In-situ samples are left out to avoid mixing disturbed and undisturbed measurements from the same location.
2. If SIP5 is enabled, replace F and ECs with the SIP5 values where the SIP3 ECs is missing and `FF_SIP5 == "Yes"`. This applies to 72 samples.
3. Fill a missing lithoclass from the last two characters of `StratLithoklasse`.
4. Drop rows that lack F, ECs, lithoclass or stratigraphy.
5. Drop `Stratigrafie == "AAOM"`.
6. Remove the `-MG` and `-WG` suffixes from the stratigraphy codes.
7. Facies scripts only: drop samples whose stratigraphy cannot be mapped to a facies (4 samples).

| | SIP3 only | SIP3 + SIP5 |
|---|---|---|
| Samples after filtering | 285 | 357 |
| … with a facies (basis of the table) | 281 | 353 |
| zm / kz / k / zg / zf / v / z / sch | 92 / 58 / 25 / 38 / 32 / 32 / 7 / 1 | 112 / 82 / 40 / 38 / 37 / 37 / 10 / 1 |

**Facies mapping of the lab samples.** This is hard-coded in `stat_analysis_ff_surfcond_lithofacies.py`:

| Facies | Stratigraphy codes |
|---|---|
| marien | NAWA, NAWO, NAZA, NAWOBE, EE, OO, MS, OOSP, BR, WAWO |
| fluviatiel | URTY, URVE, AP, BXSI, UR, PZ, EC, ST, WA, KK, KW |
| glaciaal | DRGI, DRGIGA, PENI, PE, DRUI |
| eolisch | BX, DN, BXWI, BXKO, NASC |
| organisch | NIHO, NIBA, NI |
| rest | AAOM (excluded) |

The facies of the GeoTOP and REGIS *units* comes from the "JanG" tables, not from this mapping. The two have to agree for the facies statistics to land on the right units. Units with no facies in those tables are labelled `overig` and fall back to lithoclass statistics.

---

## 3. Pipeline, step by step

All scripts are in `src/`. They run from the repository root (they `os.chdir` there themselves) and write to `data/4-output/ff_ecs_uncertainty/…`. With `SIP5 = True` the output goes to the sub-folder `SIP3_SIP5_combined/`.

### Step 0: Checking the lab results (April and August 2026)

- **`4-analyze/check_calculation_FormationFactor_SurfaceCond.py`** recomputes *F* and *ECs* for every sample from the three SIP3 measurements, σ′(1 Hz) against σw. It fits ln σ′ = ln(a σw + b), so F = 1/a and ECs = b, and writes `SIP3_check_fitted_results.xlsx` for comparison with the lab's own values.
- **`4-analyze/check_revil_results_20172026.py`** plots F–ECs per lithostrat class for the 2017 Revil data against the 2026 data, in linear and log-log space. The plots are in `FormatieFactor/revil_analysis/`.
- **Notebooks `notebooks/Revil_4point_6point_*.ipynb`** (November 2026) test whether the differences between the Revil "final report" and "paper" values come from the choice of salinity points. They compare a 4-point and a 6-point fit, the 4 highest salinities with and without the 2nd measurement, and linear fits against fits in log-space. This is QA on the reference data and does not feed the table.

### Step 1: Which lithostrat combinations matter (April 2026)

**`1-prepare/analyze_geology_flightlines.py`** rasterises the HEM flight lines onto GeoTOP (2026) and REGIS v2.2. It sums the thickness of every lithostrat combination, or every REGIS formation between −50 and −150 m, both overall and inside the saline zone (fresh–brackish interface above −150 m). It sets this against the number of lab samples per combination (`nsamples_nu.csv`). The output in `FormatieFactor/lithostrat_op_vlieglijnen/` shows where sampling and statistics matter most.

### Step 2: Exploring the data

- **`4-analyze/analyze_unc_ff_ecs.py`** (first version, March 2026): boxplots and swarm plots per lithoclass and per StratLithoklasse, plus pairwise t-tests (`pval_ff.csv`, `pval_ecs.csv`).
- **`4-analyze/cluster_analysis_{lithostrat,lithofacies,facieslitho}.py`**: scatter plots of F against ECs coloured by stratigraphy, facies or lithoclass (only groups with n ≥ 5), plus KMeans clustering on log10 F and log10 ECs with *k* chosen by silhouette score. These are exploratory and show where the groups overlap.
- **Normality.** The statistics scripts run Shapiro–Wilk tests and draw Q–Q plots and histograms for the raw and log-transformed values. Log-transformed values are much closer to normal, which is why every later step uses log-space statistics. Up to 2026-08-25 the scripts used log10. Commit `4de7647` (2026-08-26) switched to **natural log**, so that the values can go straight into `numpy.random.lognormal`. This switch is the "v4_ln" in the file name.

### Step 3: Statistics per lithoclass and per facies within a lithoclass

Script: **`4-analyze/stat_analysis_ff_surfcond_lithofacies.py`**.

1. Kruskal–Wallis tests on F and ECs:
   - by lithoclass and by facies
   - by facies within each lithoclass
   - by lithoclass within each facies

   Only groups with n ≥ 5 are tested.
2. Dunn post-hoc tests with Benjamini–Hochberg correction (`scikit_posthocs.posthoc_dunn`, α = 0.1) on the same groupings. Output: `dunn_ff_surfcond_facies_within_litho.csv`, `dunn_ff_surfcond_litho_within_facies.csv`.
3. **Manual merging** of facies whose differences were not significant. The analyst decided this from the Dunn results, and the merges are hard-coded:

   | Lithoclass | Merged facies groups |
   |---|---|
   | kz | eolisch + glaciaal; fluviatiel + marien |
   | zf | eolisch + fluviatiel; glaciaal + marien (*fluviatiel could also have gone with glaciaal + marien*) |
   | zm | eolisch + fluviatiel + glaciaal (marien stays separate) |
   | k, v, zg | one facies each with n ≥ 5 (marien, organisch, fluviatiel) |

   An alternative grouping for SIP3+SIP5 is in the script but commented out. The note "*we gebruiken deze ook voor SIP3+SIP5*" ("we use these for SIP3+SIP5 as well") records the decision to keep the SIP3 grouping, so that the SIP3 and SIP3+SIP5 tables share the same structure.
4. For each lithoclass and each facies group: n, median, mean and std of ln F and ln ECs, plus linear-scale statistics and covariances.
5. Outputs used by the look-up table:
   - `for_monte_carlo/litho_for_monte_carlo.csv` (per lithoclass)
   - `for_monte_carlo/lithofacies_for_monte_carlo.csv` (per lithoclass and facies group)
   - `for_monte_carlo/sand_clay_for_monte_carlo.csv` (all sand versus clay, for areas with less detail; not used in v4)

   The script also writes the filtered sample set with facies to `data/3-input/20260304_tbl20_WPchloride_FFdata_with_facies[_SIP3_SIP5_combined].csv`.

### Step 4: Statistics per stratigraphy within a lithoclass

Script: **`4-analyze/stat_analysis_ff_surfcond_lithostrat.py`**.

This uses the same test sequence as Step 3, with stratigraphy within lithoclass as the grouping (boxplots in `boxplots_litho_strat/`). The manual stratigraphy merges are: zg AP+PZ; kz BX+DRGI; kz NAWA+NAWO; zm NAWA+NAWO+NAZA+OO; zm BX+BXWI. The script writes `dunn_test_results_lithostrat/median_mean_std_stratlitho_manual_groups.csv`.

### Step 5: Assembling the look-up table

Script: **`4-analyze/look_up_table_monte_carlo.py`**. Set `SIP5 = True` to produce the `_SIP3_SIP5` files.

1. **Skeleton.** Concatenate the GeoTOP and REGIS unit tables (`unit_cd`, `unit_nr`, `facies`) and cross-join them with the 9 lithoclasses. The `lithoclass_id` values are a 0, v 1, k 2, kz 3, zf 5, zm 6, zg 7, g 8, sch 9.
2. **Matching unit codes to sample codes.**
   - `strat_short_name` strips a leading `NU` (`NUNAWA1` → `NAWA1`). The script stops with an error if the short names are not unique.
   - `strat_match` then strips trailing digits and trailing lower-case letters: `NAWA1` → `NAWA`, `NUECge` → `EC`, REGIS `NUBXz1` → `BX`, `NMRUBOk1` → `NMRUBO`.
3. **Fill in priority order.** A cell, once filled, is never overwritten.
   1. *Stratigraphy-specific* (`special_stats`): v-[NIHO], v-[NIBA], kz-[EC], kz-[WA], kz-[NAWA+NAWO], kz-[BX+DRGI]. These were judged to differ clearly and to make geological sense as separate groups. A cell qualifies if its lithoclass matches, its `strat_match` *starts with* the stratigraphy code, and its unit facies is one of the facies of the exactly matching unit. So `BX…` units of eolian facies qualify, but fluvial `BXSI` does not.
   2. *Facies within lithoclass*: match on (lithoclass, unit facies) against `lithofacies_for_monte_carlo.csv`.
   3. *Lithoclass*: `litho_for_monte_carlo.csv`. This covers facies `overig` and facies without their own group, for example glacial k. It keeps the Monte Carlo from failing on rare combinations.
   4. *Literature or proxy*:
      - g: F = 6.5
      - sch: F = 5.0, ECs = 0, std = 0, `dist_type = norm`
      - a: the zm lithoclass statistic (see issue 1 in §7)
4. **Split** into GeoTOP (`unit_nr == 0` or ≥ 1000) and REGIS (1 ≤ `unit_nr` < 1000). Add a second header row with units and write the CSV files.

### Version history of the tables

| File suffix | Date | Change |
|---|---|---|
| *(none)* | 2026-08-19 | First complete table; log10; SIP3; only v-[NIHO] and v-[NIBA] as stratigraphy-specific groups |
| `_FF_ECS_FRESHEM_Zeeland` | 2026-08-20 | Same skeleton with the FRESHEM-Zeeland lithoclass values (§5.3) |
| `_v2` | 2026-08-24 | Adds kz-[EC], kz-[WA], kz-[NAWA+NAWO], kz-[BX+DRGI] |
| `_v3`, `_v3_with_SIP5` | 2026-08-25 | Final facies grouping; first version with SIP5 gap filling |
| `_v4_ln_SIP3`, **`_v4_ln_SIP3_SIP5`** | 2026-08-26 | Natural log instead of log10. v4 = v3 × ln 10 (checked: ratio 2.302585) |

The Monte Carlo reads `look_up_table_ff_ECs_regis_v4_ln_SIP3_SIP5.csv` directly. For GeoTOP it reads `…_geotop_v4_ln_SIP3_SIP5_incwater.csv`, which is the same table with a row for open water added in `chloride_uncertainties`.

---

## 4. What the final tables contain

Columns: `unit_cd, unit_nr, facies, LITHOKLASSE_CD, Lithoklasse_naam, lithoclass_id, strat_short_name, strat_match, mean_dist_ff, std_dist_ff, mean_dist_surfcond, std_dist_surfcond, dist_type, statistiek_literatuur, groepering_statistiek, n`. Row 2 holds the units.

**How to sample** (from the script's own notes):

- F = exp(N(mean_dist_ff, std_dist_ff)).
- ECs [S/m] = exp(N(mean_dist_surfcond, std_dist_surfcond)). Multiply by 10 for mS/cm.
- `hem2chloride.sample_ffi_ecs25` does this with `rng.lognormal` for `lognorm` rows and `rng.normal` for `norm` rows. It treats ECs as EC25.
- F and ECs are sampled **independently**. Covariances are computed in Steps 3–4 but not carried into the table.

**Distinct distributions in the SIP3+SIP5 v4 tables.** Values are in ln-space. Geometric means (exp of the mean) are given for readability.

| Lithoclass | Group | Level | n | mean ln F | sd ln F | F (geo. mean) | mean ln ECs | sd ln ECs | ECs (geo. mean, S/m) | rows GeoTOP | rows REGIS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v | v | lithoclass | 37 | 0.768 | 0.412 | 2.16 | -3.85 | 2.61 | 0.0214 | 84 | 130 |
| v | v-[NIBA] | stratigraphy | 16 | 0.792 | 0.375 | 2.21 | -4.79 | 3.25 | 0.0083 | 1 | 0 |
| v | v-[NIHO] | stratigraphy | 12 | 0.554 | 0.278 | 1.74 | -2.82 | 2.12 | 0.0593 | 1 | 0 |
| v | v-[organisch] | facies | 31 | 0.710 | 0.353 | 2.03 | -3.95 | 2.80 | 0.0193 | 6 | 1 |
| k | k | lithoclass | 37 | 1.237 | 0.416 | 3.45 | -2.07 | 1.15 | 0.1261 | 61 | 81 |
| k | k-[marien] | facies | 31 | 1.213 | 0.401 | 3.36 | -1.81 | 0.93 | 0.1643 | 31 | 50 |
| kz | kz | lithoclass | 82 | 1.512 | 0.393 | 4.53 | -3.05 | 2.15 | 0.0473 | 12 | 7 |
| kz | kz-[BX+DRGI] | stratigraphy | 18 | 1.665 | 0.282 | 5.29 | -5.37 | 2.27 | 0.0047 | 9 | 10 |
| kz | kz-[EC] | stratigraphy | 6 | 0.728 | 0.180 | 2.07 | -3.46 | 0.59 | 0.0315 | 7 | 0 |
| kz | kz-[NAWA+NAWO] | stratigraphy | 47 | 1.507 | 0.338 | 4.51 | -1.91 | 1.00 | 0.1486 | 11 | 0 |
| kz | kz-[WA] | stratigraphy | 5 | 1.974 | 0.234 | 7.20 | -2.58 | 0.22 | 0.0759 | 1 | 3 |
| kz | kz-[eolisch+glaciaal] | facies | 19 | 1.666 | 0.274 | 5.29 | -5.57 | 2.38 | 0.0038 | 5 | 11 |
| kz | kz-[fluviatiel+marien] | facies | 62 | 1.472 | 0.412 | 4.36 | -2.18 | 1.04 | 0.1131 | 47 | 100 |
| zf | zf | lithoclass | 37 | 1.580 | 0.287 | 4.86 | -4.31 | 2.36 | 0.0134 | 12 | 7 |
| zf | zf-[eolisch+fluviatiel] | facies | 19 | 1.510 | 0.326 | 4.53 | -5.50 | 2.53 | 0.0041 | 46 | 62 |
| zf | zf-[glaciaal+marien] | facies | 18 | 1.654 | 0.226 | 5.23 | -3.06 | 1.33 | 0.0471 | 34 | 62 |
| zm | zm | lithoclass | 111 | 1.594 | 0.260 | 4.92 | -4.15 | 1.92 | 0.0158 | 12 | 7 |
| zm | zm-[eolisch+fluviatiel+glaciaal] | facies | 65 | 1.542 | 0.260 | 4.67 | -5.00 | 1.95 | 0.0067 | 49 | 74 |
| zm | zm-[marien] | facies | 46 | 1.668 | 0.243 | 5.30 | -2.94 | 1.03 | 0.0531 | 31 | 50 |
| zg | zg | lithoclass | 38 | 1.542 | 0.238 | 4.68 | -6.05 | 1.99 | 0.0024 | 57 | 78 |
| zg | zg-[fluviatiel] | facies | 30 | 1.517 | 0.200 | 4.56 | -6.04 | 1.86 | 0.0024 | 35 | 53 |
| g | literature | literature | – | 6.5 (linear) | 0 | 6.5 | 0 (linear) | 0 | 0 | 92 | 131 |
| sch | literature | literature | – | 5.0 (linear) | 0 | 5.0 | 0 (linear) | 0 | 0 | 92 | 131 |
| a | proxy = zm | proxy | 111 | 1.594 | 0.260 | (see §7) | -4.15 | 1.92 | | 92 | 131 |

What stands out:

- Marine sediments have ECs 5–10× higher than eolian and glacial sediments of the same lithoclass. This is the main reason for using the facies split.
- *F* differs much less between groups. The exception is fluvial Echteld clayey sand (kz-[EC], F ≈ 2.1).
- The ln ECs spreads are wide (σ ≈ 2–3 for several groups). This is partly because 31 of the 357 samples have ECs at the reporting floor of 1·10⁻⁴ S/m (§7, issue 6).
- REGIS has no Holocene peat units and no EC or NAWA units, so v-[NIHO], v-[NIBA], kz-[EC] and kz-[NAWA+NAWO] occur only in GeoTOP.

---

## 5. Side avenues

### 5.1 Shells and gravel: Monte Carlo propagation through Archie's law

Script: `1-prepare/mc_propagation_shellsgravel.py` (Joost Delsman, Aug–Sep 2026; Romee van Dam, revised Oct 2026 as "new method").

There are almost no lab samples of shell and gravel layers (one shell sample, OOSP, F = 5.39). F is therefore derived from **Archie, F = φ^(−m)**, with uncertain porosity φ and cementation exponent m:

- **Porosity.** Measured ranges from Ronald: shells φ = 0.36–0.502 (n = 14), gravel φ = 0.245–0.346 (n = 16). The range is turned into a lognormal distribution: μ is the geometric midpoint, and σ = ln(max/min) / dₙ, where dₙ is the expected range of *n* standard-normal draws (computed by numerical integration). An earlier version also tried Blom's approximation, which gives nearly the same result.
- **Cementation exponent m.**
  - *Old method (Aug–Sep):* literature ranges suggested by Copilot, shells m = 1.7–3.0 and gravel m = 1.4–1.8, treated in the same way as porosity.
  - *New method (Oct, current code):* a lognormal fitted directly to the 156 measured m values of zf, zm and zg samples (`CementationExponent_m_unitless` from `tbl05_Measurementdata_Full`; median 1.59, range 0.95–2.86). The same distribution is used for both shells and gravel.
- 10⁶ draws, then a lognormal fit with loc = 0 to the resulting F.

| | Old method (`FF_shells_gravel`, 14-09-2026) | New method (rerun 27-11-2026) |
|---|---|---|
| Shells | median F 5.0, σ_ln 0.6 | median F **4.15**, σ_ln 0.31 (p5–p95: 2.6–7.2) |
| Gravel | median F 5.3, σ_ln 0.3 | median F **7.79**, σ_ln 0.41 (p5–p95: 4.3–16.2) |

**Status.** The v4 tables still use the fixed values F = 6.5 (gravel) and 5.0 (shells) with zero spread and ECs = 0. The new-method distributions are ready to add as `lognorm` rows: shells μ_ln 1.42 / σ_ln 0.31, gravel μ_ln 2.05 / σ_ln 0.41.

Points to discuss:

- Using sand m for shells and gravel is an assumption. Shells in particular may have a higher m.
- The input path in the script points to a personal OneDrive folder, and the script writes no output file other than the CSV.

### 5.2 Boom Clay (Boomse klei, REGIS `NMRUBOk1`): F from HEM resistivity

Script: `1-prepare/boomseklei.py` (Joost Delsman, Sep 2026). There are no lab samples of Boom Clay. The method:

1. Select the Monte Carlo voxels in REGIS `NMRUBOk1` in pilot area Zeeland Zuid C3 (`Results_Monte_Carlo/ZLZ-C3_28-08-26`). Shift REGIS by 25 m to align it with the HEM grid, match each voxel to the nearest HEM depth (cached in `data/4-output/boomseklei/rubo_regis_xy.parquet`), and merge with the HEM resistivity (`sharp_rho_mean`).
2. Assume saline pore water of 16 g/L Cl. Using Cl = 0.36·EC25 − 0.45, this gives EC25 = 45.7 mS/cm, or 32.9 mS/cm at 11 °C (factor 1.39).
3. Compute the apparent formation factor F_app = ECw / ECb (ECs = 0), then a corrected F = ECw / (ECb − ECs).
4. **ECs from literature.** Boom Clay is rich in smectite with a CEC of about 15–30 meq/100 g (Zeelmaekers et al., 2015). ECs is taken as 0.01–0.05 S/m, read as the 5th–95th percentile of a lognormal (μ_ln −3.80, σ_ln 0.49, median 0.022 S/m).
5. **Consistency check.** With porosity ≈ 0.4 (from about 30 % water content, Deltares report 11211277-003-GEO-0001), F ≈ 9 implies m ≈ 2.5, a normal value for clay.
6. **Monte Carlo.** 10⁶ draws: ECb from a lognormal fitted to the voxel resistivities and ECs from the expert distribution. Compute F and keep 0 < F < 50.

Results (rerun 27-11-2026, 86,888 voxels, median ρ = 2.61 Ωm):

| | Value |
|---|---|
| F_app (ECs = 0) | 8.6 |
| F with ECs = 0.025 S/m | 9.0 |
| Monte Carlo F | median 9.8, p5–p95 5.4–23.7; lognormal with loc = 0: μ_ln 2.33, σ_ln 0.45 |

**Status.** This is not in the tables. In the REGIS v4 table `NMRUBOk1` × k gets `k-[marien]`, F ≈ 3.4 (geometric mean) and ECs ≈ 0.16 S/m. That is about a third of the Boom Clay estimate, so chloride in Boom Clay is probably overestimated by about 2.5×.

The script has three defects:

- The `# TODO: schrijf parameters weg` ("write parameters out") was never done, so no output file is written.
- `ecs_shape, ecs_loc, ecs_scale = dist_ecs` raises an error, because a frozen scipy distribution cannot be unpacked.
- When the cached parquet exists, `gdf_resis["z"]` is **not** rounded before the merge. The merge then keeps only 12 voxels instead of about 87,000. This is fixed by moving the rounding outside the `try` block.

The resistivity fit is also a free 3-parameter `lognorm.fit` (loc ≠ 0). The comment "*Waarom zo vreemd hoog?*" ("why so strangely high?", about F_app = 8) is still an open question. Possible causes: lower actual salinity, HEM smoothing over thin layers, or genuinely low porosity.

### 5.3 FRESHEM-Zeeland look-up table

`4-analyze/look_up_table_monte_carlo_FF_ECs_FRESHEM_zeeland.py` fills the same skeleton with the lithoclass values used in FRESHEM Zeeland (`geotop_lithoclass_props_oude_FF_ECs.csv`; ECs converted from mS/cm to S/m). These values are linear, with zero spread. The output is `look_up_table_ff_ECs_{geotop,regis}_FF_ECS_FRESHEM_Zeeland.csv` (2026-08-20). It serves as a reference run, so that the effect of the new statistics can be isolated. The same Zeeland values were used for the preliminary fresh–salt sections for the regional meetings (`4-analyze/preliminary_results.py`, ECs per lithoclass in mS/cm hard-coded).

### 5.4 Dependence between F and ECs: copulas

`4-analyze/fit_bivariate_copulas_three_marginals_ff_surfcond_lithofacies.py` (July 2026) fits bivariate copulas (`pyvinecopulib`) to (F, ECs) per lithoclass and per facies group within a lithoclass. It tries three marginal choices: empirical, lognormal and normal. Output goes to `copula_results_three_marginals[_low_values_corrected]/`. The point was to test whether F and ECs should be drawn jointly. **It was not carried into the table**: the Monte Carlo draws F and ECs independently. The covariances (`cov_log_ff_surfcond`) are available in the `median_mean_std_*` outputs if joint sampling is wanted later.

### 5.5 Statistics on 1/F

`4-analyze/stat_analysis_inverse_ff_surfcond_{facies,lithostrat}.py` (July 2026) repeats the Kruskal–Wallis and Dunn analysis on 1/F. 1/F is the slope of ECb against ECw and enters the conversion linearly. This was a sensitivity check of the grouping. Rank-based tests on F and 1/F give the same p-values, and the v4 table uses F.

### 5.6 Direct ECbulk → chloride (November 2026)

`src/formationfactor_from_xyzdata/` tests skipping F and ECs altogether by fitting a direct bulk EC → chloride relation to paired HEM and chloride data, generic and formation-aware. It includes an interactive explorer (`explorer.html`). The comparison with hem2chloride is reported separately in `reports/direct_ecbulk_chloride/direct_ECbulk_chloride_test.pdf`.

### 5.7 Small utilities

- `1-prepare/test_pdf_resis.py` and `4-analyze/test_np_input_MSD.py` check how scipy and numpy parameterise lognormals: `s`/`scale` against μ/σ against GM/MSD, and ln against log10. These checks led to the switch to ln.
- `5-visualize/topomap_results.py` makes a topographic map for a transect.
- `5-visualize/ff_ecs_groups.py` and `plot_ff_ecs_groups.py` (not yet committed) rebuild and draw the final groups.

---

## 6. Reproducibility

**Run order** (from the repository root; set `SIP5 = True` in each script):

1. `src/4-analyze/stat_analysis_ff_surfcond_lithofacies.py` → `litho_for_monte_carlo.csv`, `lithofacies_for_monte_carlo.csv`
2. `src/4-analyze/stat_analysis_ff_surfcond_lithostrat.py` → `median_mean_std_stratlitho_manual_groups.csv`
3. `src/4-analyze/look_up_table_monte_carlo.py` → the six `look_up_table_ff_ECs*_v4_ln_SIP3_SIP5.csv` files
4. Copy to `Nextcloud/…/ff_ecs_uncertainty/for_monte_carlo/`. For GeoTOP, add the water row in `chloride_uncertainties` to make `…_incwater.csv`.

The code currently has `SIP5 = True` in both stat scripts and `SIP5 = False` in `look_up_table_monte_carlo.py`. The cluster scripts have `SIP5 = False`.

**Verification (2026-11-27).** I loaded the lab table with the same filters (`src/5-visualize/ff_ecs_groups.py`) and recomputed:

- the lithoclass ln-statistics for SIP3+SIP5, and
- the facies groups and stratigraphy-specific groups with their sizes.

All lithoclass means, standard deviations and n agree with the v4 SIP3+SIP5 tables to full precision; for example k: n = 37, 1.237396 / 0.416335 / −2.070970 / 1.149843. All group sizes agree too. The v4 SIP3 and SIP3+SIP5 tables share the same row keys.

**Gaps that block a one-click rebuild:**

- The intermediate SIP3+SIP5 outputs (`SIP3_SIP5_combined/…`) exist only on the machine where they were produced. The `for_monte_carlo` folder on Nextcloud holds only the older SIP3, log10 intermediate files of 2026-08-18.
- The two "JanG" unit→facies CSV files are not in the local `data/1-external`.
- Hard-coded personal paths: `mc_propagation_shellsgravel.py` points to OneDrive, `check_calculation_…py` uses `os.chdir`, and `boomseklei.py` points to Nextcloud.
- The SIP5 switch must be set by hand in three scripts. Nothing checks that they agree.

---

## 7. Issues and recommendations

1. **Anthropogenic rows (`a`, lithoclass_id 0) are sampled wrongly in the published v4 tables.** Step 4 copies the zm **ln-space** statistics but labels them `dist_type = norm`. `hem2chloride.sample_ffi_ecs25` then draws F ~ N(1.59, 0.26), so F ≈ 1.6 instead of about 4.9, and ECs ~ N(−4.15, 1.92), which is mostly **negative**. **Fixed in the code (2026-11-27):** In `look_up_table_monte_carlo.py`, `a` now copies the generic lithoclass row of `replacement_litho` (zm) in full, `dist_type` included, so it follows zm automatically. g and sch keep `norm` for their linear literature values. *The published CSV files still contain the error* until the script is rerun with the SIP3+SIP5 intermediate files. As a stopgap, the `a` rows can be edited by hand to `lognorm`.
2. **Gravel and shells have fixed values with zero spread, and ECs = 0.** Replace them with the §5.1 new-method distributions (lognorm; shells μ_ln 1.42 / σ_ln 0.31, gravel μ_ln 2.05 / σ_ln 0.41). ECs = 0 is reasonable for gravel. For shells it is an assumption: the one OOSP sample has ECs ≈ 0.012 S/m.
3. **Boom Clay.** Add a REGIS-specific row for `NMRUBOk1` × k from §5.2 (F μ_ln ≈ 2.3 / σ_ln ≈ 0.45; ECs μ_ln −3.80 / σ_ln 0.49) after the open question about F_app has been discussed. First fix the z-rounding and output defects in `boomseklei.py`.
4. **Deterministic mode in the consumer.** With `rng=None` (non-Monte Carlo runs), `sample_ffi_ecs25` used the `mean_*` column as is, so for `lognorm` rows ln F (≈ 1.5) was used as F. **Fixed in `chloride_uncertainties` (2026-11-27):** the deterministic branch now depends on `dist_type` from the input file. `norm` rows use the mean. `lognorm` rows use the median exp(mean_ln), not the lognormal mean, which explodes for σ_ln ECs ≈ 2–3. A regression test was added.
5. **Independent sampling of F and ECs.** This is acceptable as a first step. The copula analysis (§5.4) and the stored covariances make joint sampling possible if the dependence turns out to matter for chloride uncertainty.
6. **ECs floor values.** 31 samples have ECs = 1·10⁻⁴ S/m, the reporting floor, mostly in zg, kz, v, zf and zm. In ln-space they pull the mean down and inflate σ_ln ECs (for example zg σ_ln = 2.0). Consider treating them as censored values or excluding them (compare `copula_results_…_low_values_corrected`), and document the choice.
7. **Grouping choices are partly manual.** The merges and `special_stats` are expert decisions made with α = 0.1 and n ≥ 5, and some groups are small (kz-[WA] n = 5, kz-[EC] n = 6). For SIP3+SIP5 the SIP3 grouping was kept on purpose. Record the Dunn p-values behind each merge next to the table.
8. **Housekeeping.**
   - Put the SIP5 switch in one shared config.
   - Remove the personal paths.
   - Publish the `SIP3_SIP5_combined/for_monte_carlo` intermediate files and the JanG facies CSV files to Nextcloud next to the final tables.
   - Commit `src/5-visualize/ff_ecs_groups.py` and `plot_ff_ecs_groups.py`. `ff_ecs_groups.py` doubles as a regression check of the group structure.
