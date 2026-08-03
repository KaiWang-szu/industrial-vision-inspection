# Industrial Vision Inspection

An end-to-end classical computer-vision pipeline for sub-pixel dimensional measurement, product-specific defect inspection, and confidence-aware routing of previously unseen inputs.

> Course project for *Machine Vision Technology and Industrial Applications*. The public repository contains code, evaluation summaries, and representative figures; the original industrial image set is intentionally excluded.

![Pipeline overview](assets/algorithm_overview_clean.png)

## Highlights

- **Sub-pixel measurement** — Otsu segmentation, connected-component filtering, pose correction, gray-level interpolation, and robust line fitting.
- **Product-aware inspection** — separate, interpretable detectors for scratch-like anomalies and edge-band defects.
- **Unknown-product routing** — builds a normal-sample model library, estimates product identity and confidence, then routes uncertain inputs to manual review.
- **Auditable outputs** — CSV/JSON metrics, overlays, distributions, and single-image prediction artifacts.

## Results

| Evaluation track | Key result |
|---|---:|
| Product 1 image-level inspection | Accuracy **96.70%**, F1 **96.39%** |
| Product 2 image-level inspection | Recall **91.11%**, F1 **73.21%** |
| Unknown-product identification | Overall accuracy **91.15%** |
| Confidence-aware automatic routing | Coverage **91.67%** |
| Identification among accepted samples | Accuracy **99.43%** |
| Low-confidence review queue | **8.33%** of inputs |
| End-to-end unknown-product defect detection | Recall **94.19%**, F1 **81.82%** |

The image-level and pixel-level objectives behave differently: the current detector is strong at screening defective products, while precise mask localization remains the main area for improvement. Full machine-readable results are in [`results/`](results/).

![Unknown-product framework](assets/unknown_product_framework_clean.png)

## Method

### 1. Dimensional measurement

The pipeline segments the dark workpiece, keeps the largest connected component, estimates its orientation with a minimum-area rectangle, and aligns the long axis horizontally. Four edges are localized with threshold-crossing interpolation and fitted independently. It reports width, height, edge RMS, and a rectangularity score.

![Measurement flow](assets/measurement_flowchart_clean.png)

### 2. Defect inspection

- **Product 1:** template difference, top-hat/black-hat responses, local-background residuals, and non-horizontal Hough evidence target scratches and local texture anomalies.
- **Product 2:** an edge-region prior combines template residuals, row-background subtraction, and local intensity deviations for wide band-shaped defects.

Morphology, component area, and shape constraints turn candidate pixels into image-level OK/NG decisions.

![Defect overlay comparison](assets/defect_overlay_comparison_revised.png)

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python src/industrial_inspection.py --data-root data --output outputs
```

For one incoming image:

```bash
python src/industrial_inspection.py \
  --data-root data \
  --output outputs \
  --predict-image path/to/query.bmp
```

See [`data/README.md`](data/README.md) for the expected private-data layout.

## Repository layout

```text
.
├── src/industrial_inspection.py   # complete experiment and CLI
├── assets/                        # selected architecture/result figures
├── results/                       # compact evaluation summaries
├── data/README.md                 # input contract; no raw data
└── requirements.txt
```

## Limitations and next steps

- Thresholds are calibrated on two product families and should be re-estimated under new illumination or camera settings.
- High screening recall comes with fragmented pixel masks, especially for Product 2.
- A stronger deployment version would add illumination normalization, component-level calibration, and a learned anomaly branch while retaining the review mechanism.

## Author

Kai Wang · Electronic Information Engineering, Shenzhen University
