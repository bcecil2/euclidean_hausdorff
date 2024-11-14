# Changelog

<!--next-version-placeholder-->

### v1.3.1 (Oct 25 2024)

### Feature

- New signature for upper(): n_dH_iter and n_err_ub_iter instead of max_n_iter and dH_iter_share

### v1.3.0 (Oct 25 2024)

### Feature

- New signature for upper(): max_n_iter and dH_iter_share instead of n_dH_iter and n_err_ub_iter
- Only pruning if there was a change to best dH

### Fix

- Fixed issue occurring when pruning fully empties the grid

### v1.2.5 (Oct 24 2024)

### Feature

- Refactored upper()

### Fix

- Fixed missing target_err stopping condition

# v1.2.4 (Oct 23 2024)

### Feature

- Removed storing grid points whose cells cannot improve on the currently best dH
- Refactored the tests

### Fix

- Fixed application of translations

## v1.2.3 (Oct 20 2024)

### Feature

- Showing zoomed-in grid points when verbose > 2 

### Fix

- Error bound calculation now accounts for dH-minimizing iterations

## v1.2.2 (Sep 20 2024)

### Feature

- Showing timestamps in performance tests 

### Fix

- Possible dH used for sorting grid points in error-minimizing iterations is not thresholded by 0 from below anymore

## v1.2.1 (Sep 15 2024)

### Feature

- Reordered iterations order: error-minimizing iterations are now performed before dH-minimizing iterations

### Fix

- Uniformized performance tests

## v1.2.0 (Sep 13 2024)

### Feature

- Changed upper() signature to use n_dH_iter (the number of dH-minimizing iterations) and n_err_ub_iter (the number of error-minimizing iterations) 

## v1.1.2 (Sep 2 2024)

### Feature

- Added performance tests in 2D for p=5

### Fix

- Fixed issue with the initial cell not containing the entire ball

### Documentation

- Added this changelog
