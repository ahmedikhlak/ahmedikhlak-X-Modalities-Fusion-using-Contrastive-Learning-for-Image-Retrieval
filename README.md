# XM Framework: Contrastive Learning for Composed Image Retrieval

This repository provides the implementation of the proposed X-Modalities (XM) framework for composed image retrieval using CLIP-based alignment, pseudo-token enhancement, and relational network fusion.

---

## 📦 Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- faiss-gpu
- pandas, numpy
- tqdm
- comet_ml (optional)
- OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`)

---

## 🔧 Setup Instructions

1. Clone the repository:
    ```bash
    git clone <your_repo_url>
    cd <your_repo_name>
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # or `env\Scripts\activate` on Windows
    ```

<!-- 3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ``` -->

---

## 📁 Dataset Preparation

- **FashionIQ:** Download and structure the dataset as per [FashionIQ dataset format](https://github.com/XiaoxiaoGuo/fashion-iq).
- **CIRR:** Prepare CIRR dataset following the [CIRR setup guide](https://github.com/CIRR-dataset).

Place both datasets in a `./data` directory.

---

## 🚀 Run the Pipeline

The complete experimental flow is orchestrated by:

```bash
python pipeline.py
```

This script sequentially:
- Trains the relational network (combiner)
- Fine-tunes CLIP on FashionIQ captions
- Runs inference on CIRR test set
- Evaluates the model on FashionIQ and CIRR

---

## 📊 Output

- Logs and checkpoints are saved to `./logs` and `./checkpoints`.
- Final predictions are written to `./results`.

---

## 🧪 Evaluation Metrics

- Recall@K
- Rank Accuracy
- Feature Similarity

---

## 🧑‍💻 Citation

If you use this code in your research, please cite our paper:

```
@article{Ahmed2024XM,
  title={X-Modalities Fusion using Contrastive Learning for Image Retrieval},
  author={Ikhlaq Ahmed et al.},
  journal={PeerJ Computer Science},
  year={2024}
}
```

---

For questions or issues, feel free to contact the corresponding author.