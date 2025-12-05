# Alarm_Flood_Classification_Explainable_Bifurcation_Points

This repository contains research code for combining conformal prediction with counterfactual explanations in the context of online alarm flood classification in industrial process plants.
The framework detects bifurcation points (moments when the conformal prediction set shrinks) and explains them via counterfactual alarm patterns.

## Main Components
- `ConformalAlarmClassifier`: wraps an alarm flood classifier and provides step-wise conformal prediction sets with a set-level delay timer to suppress chattering bifurcations.
- `BifurcationDetector`: scans conformal prediction sets over time and identifies bifurcation events per sample.
- `CounterfactualGenerator`: generates counterfactual alarm floods at bifurcation points using different strategies (`random`, `all`, `greedy`).
- `compute_counterfactual_metrics`: evaluates the quality and behaviour of generated counterfactuals (e.g. number of swapped variables, cost, distances, overlap with ground truth).
- Synthetic ground-truth structures (`ground_truth_bifurcations`, `per_class_bifurcations`) describe ideal class ambiguities and alarm-variable relevance over time.

## Key Ideas
- Online alarm flood classification is performed step-wise (expanding window) using existing AFC models (e.g. CASIM, ACM-SVM, EAC-based classifiers).
- Conformal prediction produces time-resolved prediction sets for each alarm flood; a delay timer ensures that small oscillations do not immediately create new bifurcation points.
- A bifurcation point is defined as the first time step at which the size of the accepted prediction set decreases and remains changed after the delay timer has elapsed.
- At each bifurcation and for each dropped class, the `CounterfactualGenerator` searches for a minimal change in alarm variables that would have kept the class in the conformal prediction set.
- The counterfactual search is guided by a cost function: `cost = shortfall + λ * distance`, where shortfall measures how far the candidate is from satisfying the conformal threshold and the distance term penalizes deviation from the original alarm flood.

## Counterfactual Strategies
- `random`: randomly selects a subset of alarm variables from the nearest neighbour and swaps them into the original alarm flood (single attempt).
- `all`: swaps all alarm variables from the nearest neighbour (up to the bifurcation time).
- `greedy`: iteratively adds alarm variables that differ between neighbour and original, selecting at each step the variable set that minimizes the cost until the target class enters the conformal prediction set.
- For `random` and `all`, only the single closest calibration sample from the target class is used.
- For `greedy`, multiple nearest neighbours (up to `k_neighbors`) are considered and the best counterfactual across them is selected.

## Nearest Neighbour Selection
- Nearest neighbours are taken from the calibration set restricted to the target class.
- Hamming distance is used between the original alarm flood and calibration samples, from time 0 up to the current bifurcation time step.
- Only neighbours for which the target class is already in the conformal prediction set (for the neighbour itself at the bifurcation step) are eligible; distances and rankings are computed on this filtered subset.

## Metrics and Evaluation
- For classification and conformal prediction:
  - Accuracy and F1-score at the final step.
  - Average prediction set size and coverage per step.
  - Comparison with theoretically expected average set size derived from `ground_truth_bifurcations`.
- For bifurcations:
  - Total and average number of detected bifurcations per fold and classifier.
  - Comparison with expected number of bifurcation points based on per-class ground-truth counts.
- For counterfactuals:
  - Fraction of events where a counterfactual was found.
  - Number of swapped variables.
  - Attempts (for greedy search).
  - Cost, distances to original and neighbour.
  - Overlap of swapped alarm variables with ground-truth relevant variables at the corresponding bifurcation.

## Usage Outline
1. Prepare your alarm flood dataset `X` (shape: `n_samples x n_alarms x n_steps`) and labels `y`.
2. Define base classifiers and hyperparameters (e.g. CASIM, ACM-SVM, EAC-based classifiers).
3. Instantiate `ConformalAlarmClassifier` with chosen classifier, `step_list`, and conformal parameters (`alpha`, `delay_timer`).
4. Use cross-validation:
   - Split into train, calibration, and test sets.
   - Fit `ConformalAlarmClassifier` and compute conformal sets on test data.
   - Run `BifurcationDetector` to extract bifurcation events.
   - For each strategy (`random`, `all`, `greedy`), instantiate `CounterfactualGenerator` and generate counterfactuals for test samples.
5. Aggregate and evaluate results using `compute_counterfactual_metrics` and the provided summary routines.
6. Optionally export detailed counterfactuals and metric summaries to CSV for further analysis and plotting.

## Dependencies
- Python 3.x
- NumPy
- pandas
- scikit-learn
- joblib
- tqdm
- Existing AFC classifiers and utilities from your codebase (e.g. CASIM, ACM_SVM, EAC-based models, alarm conversion utilities).