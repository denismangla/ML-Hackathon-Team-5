# Data notes, assumptions and decisions (P1)

Figures below came from one run of `python -m src.eda`, `build_dataset` and `coverage`.
**Re-run them yourself and check these numbers before putting any in the report.**

## How the data is organised
* 90 trips with labels: 80 `train`, 10 `student_test` (208,417 and 19,160 GPS points).
* Each trip has its **own** road network (up to ~155k nodes) and is in a different place:
  25 distinct UTM zones across the 90 trips. A single global projection would be wrong, so
  `io_utils.load_trip` picks the zone from each trip's GPS centroid.
* Columns are `lon, lat` in that order (e.g. trip 00000001 spans lon 8.57-8.65, lat 49.84-50.21).
* GPS is ~1 Hz (median 99.1% of steps exactly 1 s). No missing values, no non-positive time steps.

## Labelling proposal (agree with the team early)
`.route` is an ordered arc list for the whole trip, not one arc per point. Each GPS point is
assigned a route arc by a monotone dynamic programme minimising point-to-arc distance
(`src/labels.py`, skip penalty 2 m per skipped arc). A point's label is trusted
(`label_ok = 1`) if it lies within 50 m of its assigned arc. Verified visually
(`results/figures/label_alignment_check.png`).

## Data-quality issues found
1. **Reference route does not always describe the driven path.** In 5/80 train trips under 95%
   of points are within 50 m of their route arc; trip 00000054 has only 10.6% and 00000033 41%.
   Example (trip 00000002): the vehicle travels km at ~7 m/s inside a few-hundred-metre area
   while `.route` lists each arc once. 10 trips have repeated arcs and 11 have small gaps
   in consecutive-arc connectivity (>=98%). I cannot confirm the cause from the files alone.
   -> `label_ok` flag; overall 88.8% of train points are trusted, 99.9% of student_test.
2. **Both directions of a road exist as separate arcs** at the same distance. Nearest-arc
   accuracy is therefore only ~54% (89% if the reverse twin is accepted); heading features
   resolve this (a crude distance+heading rule already reaches ~83% on student_test).
3. **Time gaps**: 47/90 trips have gaps > 5 s; extreme ones in train trips 00000090
   (153,064 s) and 00000078 (52,810 s). Handle in the HMM.
4. **GPS jumps** (implied speed > 60 m/s) in 20/90 trips, up to 365 steps in trip 00000032.
5. **Road networks are fragmented**: 89/90 have >1 connected component (median 19, max 421),
   but the largest component holds a median 98% of nodes (min 69%). No isolated nodes.

## Candidate settings and coverage (train, trusted labels)
Radius 100 m, K = 20. The true arc is missing from the candidates for 0.29% of trusted
train points (0.01% on student_test). K = 10 would give 0.65%.
Caveat: because trusted labels are by definition <= 50 m from their arc, the radius table
cannot show anything above 50 m; the 10.6% overall train miss rate is almost entirely
untrusted-label points.
