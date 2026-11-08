# Content-Based Satellite Image Retrieval (UCMerced Land Use)

Project for the Information Retrieval course (A.Y. 2025/2026), task: Content-Based Image Retrieval (CBIR).

## What the project does
Given a query satellite image, the system retrieves the most similar images from a database.
It compares two pretrained feature extractors (ResNet50, CLIP ViT-B/32), a fusion of both,
and four FAISS indexes (exact, HNSW, IVF-PQ, binary hash codes).
Relevance is defined by the land-use class label.

## Environment
- Python: 3.13.16
- GPU: Tesla T4 (Google Colab)
- Libraries: see `requirements.txt`

```
pip install -r requirements.txt
```

## Dataset (not included, about 2 GB)
Kaggle version of UCMerced Land Use: https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification

This version has 21 classes with 500 images each (10,500 total), which differs from the
standard 2,100-image benchmark. Only the folder `images/` is used. Download and unzip it so that
the path is `/content/ucmerced/images/<class_name>/*.png` (or change `root` in the notebook).

Download inside Colab (needs a Kaggle API token):
```
!kaggle datasets download -d apollo2506/landuse-scene-classification -p /content
!unzip -q /content/landuse-scene-classification.zip -d /content/ucmerced
```

## How to run
1. Open `notebooks/cbir_pipeline.ipynb` in Google Colab (GPU runtime recommended).
2. Install the requirements and download the dataset as above.
3. Run the cells in order. The notebook:
   - checks for near-duplicate images,
   - extracts ResNet50 and CLIP embeddings (L2-normalized),
   - splits the data (seed 42): 50 queries and 450 database images per class
     (1,050 queries, 9,450 database images),
   - evaluates P@k and mAP,
   - fuses the two models by weighted concatenation (w = 0.5),
   - compares FAISS indexes and saves tables and plots to `results/`.

## Main results (single random split)
| Model | P@1 | P@10 | mAP (full ranking) |
|---|---|---|---|
| ResNet50 | 0.9914 | 0.8872 | 0.5046 |
| CLIP ViT-B/32 | 0.9733 | 0.8939 | 0.5283 |
| Fused (w = 0.5) | 0.9981 | 0.9310 | 0.5748 |

| Index | P@10 | Top-10 overlap with exact | Recall@100 vs exact | Bytes per vector |
|---|---|---|---|---|
| Flat (exact) | 0.9310 | n/a | 1.0 | 10,240 |
| HNSW | 0.9310 | 1.00 | 0.9999 | 10,240 |
| IVF-PQ | 0.9106 | 0.70 | 0.7863 | 64 |
| Binary hash | 0.8868 | 0.53 | 0.4991 | 320 |

## Limitations
- One random split only; small differences are not statistically established.
- Fusion weights were explored on the evaluation queries, so tuned values are slightly optimistic.
- Results are not comparable with published numbers (different dataset version and split).
- Timings at this scale are noisy and are not used for conclusions.
- Relevance is defined only by class label.

## Repository structure
```
notebooks/cbir_pipeline.ipynb   main notebook with outputs
results/                        CSV tables and PNG plots
requirements.txt                library versions
README.md                       this file
```
