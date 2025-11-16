# SLI‑Word‑Level

SLI (Sign Language Interpreter) is a deep learning-based system for real-time sign language translation at the word level. The project leverages MediaPipe for pose and hand landmark extraction and explores multiple deep learning architectures, including transformers and LSTMs with attention mechanisms. Trying different approaches, we achieved a model with 90% test accuracy.

---

## Demo Video
A short demo showing the system in action:  
https://drive.google.com/file/d/1Y_bmxzzyLHVoqKNX6o354n5tK4ZCYBr2/view?usp=sharing

---

## Dataset  
Base dataset (WLASL) used for pretraining:  
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

Due to the absence of a public Egyptian Sign Language (ESL) dataset, we would appreciate any contribution  
If you would like to **contribute ESL recordings**, contact:  
**esl.dataset@gmail.com**

---

## Full Research Paper
The complete updated research paper is available in:  
`docs/ResearchPaper.pdf`

---

## Important
This repository provides:

- The **real‑time inference system**
- Feature extraction utilities using **MediaPipe**
- Optional grammar correction using **Sapling API**
- The complete **research paper**

Training scripts are **not included**, but all model architectures and hyperparameters are fully detailed in the paper.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/muha-0/SLI-Word-Level.git
cd SLI-Word-Level
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the Trained Model
Place the trained model at:

```
models/SLI_V2.keras
```

Or specify a custom path using:

```bash
export SLI_MODEL_PATH="/path/to/model.keras"
```

### 4. (Optional) Enable Grammar Correction
```bash
export SAPLING_API_KEY="your_key_here"
```

### 5. Run the Real‑Time Interpreter
```bash
python src/realtime_inference.py
```

A webcam window will open and predictions will appear as you sign.  
Dropping your hands signals the end of a sign, triggering inference.  
Press **Q** to exit.

---

## Project Structure
```
SLI-Word-Level/
│
├── docs/                       # Research paper
├── models/                     # Saved trained model
├── src/
│   ├── realtime_inference.py   # Main interpreter
│   ├── sapling_api.py          # Grammar correction (optional)
│   └── utils/                  # Feature extraction and helpers
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Team Members
- Nour Hany  
- Laila Khaled  
- Yasmine Mohamed  
- **Ahmed Sameh**

---

## License
This project is released under the **MIT License**.  
Refer to the `LICENSE` file for details.
