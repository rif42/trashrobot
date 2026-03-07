# Fix Bbox Format - Direct JSON Edit

## Task
Fix the bbox variable assignment order in train_yolo_local.ipynb from:
```python
y_center, x_center, height, width = bbox
```
to:
```python
x_center, y_center, width, height = bbox
```

There are 2 occurrences:
1. Cell 3 (visualization) - around line 133
2. Cell 4 (label generation) - around line 175

Also update the comments to say "Standard YOLO format" instead of "HaGRID format".

## Files
- train_yolo_local.ipynb

## Changes

### Change 1: Cell 3 Visualization
Find:
```
            # HaGRID format: [y_center, x_center, height, width]
            y_center, x_center, height, width = bbox
```

Replace with:
```
            # Standard YOLO format: [x_center, y_center, width, height]
            x_center, y_center, width, height = bbox
```

### Change 2: Cell 4 Label Generation
Find:
```
            # HaGRID format: [y_center, x_center, height, width] - already normalized
            y_center, x_center, height, width = bbox
```

Replace with:
```
            # Standard YOLO format: [x_center, y_center, width, height] - already normalized
            x_center, y_center, width, height = bbox
```
