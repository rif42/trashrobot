# Fix HaGRID Coordinate Format in Notebook

## Summary
The HaGRID JSON annotations use a non-standard format `[y_center, x_center, height, width]` instead of the standard YOLO `[x_center, y_center, width, height]`. This causes bounding boxes to be drawn in the wrong location.

## Problem
The current code assumes standard YOLO format:
```python
x_center, y_center, width, height = bbox  # WRONG!
```

But HaGRID uses:
```python
y_center, x_center, height, width = bbox  # CORRECT!
```

## Changes Required

### Fix 1: Cell 3 Visualization Code
**File**: `train_yolo_local.ipynb` (Cell 3, around line 200)

Change:
```python
x_center, y_center, width, height = bbox
```

To:
```python
y_center, x_center, height, width = bbox
```

### Fix 2: Cell 2 Label Generation  
**File**: `train_yolo_local.ipynb` (Cell 2, in the `create_yolo_labels` function)

The label generation code also needs to be fixed to parse the bbox correctly:

Current (WRONG):
```python
for bbox, label in zip(bboxes, labels):
    if label == 'ok':
        x_center, y_center, width, height = bbox
        ok_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
```

Fixed (CORRECT):
```python
for bbox, label in zip(bboxes, labels):
    if label == 'ok':
        y_center, x_center, height, width = bbox
        ok_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
```

## Verification
After fixing, the visualization should show red bounding boxes properly around the "ok" hand gestures, not in incorrect locations.

## Technical Details
The HaGRID dataset stores bboxes as:
- `[0.5097, 0.2832, 0.0734, 0.1500]` 
- Means: y_center=0.51, x_center=0.28, height=0.073, width=0.15
- Not: x_center=0.51, y_center=0.28, width=0.073, height=0.15

This is a dataset-specific quirk that must be accounted for when converting to standard YOLO format.
