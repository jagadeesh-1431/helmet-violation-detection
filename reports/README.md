# Reports

This folder holds evaluation output that the scripts **generate from measured results**:

| File | Produced by | Contains |
|------|-------------|----------|
| `model_evaluation.md` | `python scripts/evaluate.py ...` | per-class precision, recall, mAP@50, mAP@50-95, TP/FP/FN at the app's default confidence, confusion matrix, speed and hardware |
| `contact_sheet*.jpg` + `.txt` | `python scripts/visualize_dataset.py ...` | dataset images with their labels drawn on, and the list of files shown |
| `eval_plots/` | `scripts/evaluate.py` | Ultralytics plots (confusion matrix, PR curves) |

Nothing here is written by hand and nothing is shipped pre-filled. If a file is missing, the measurement has not been
run yet. Quote numbers from `model_evaluation.md` only together with the split, dataset and weights file it names.

Images are not committed by default (see `.gitignore`) because they may contain dataset pictures with their own licence.
