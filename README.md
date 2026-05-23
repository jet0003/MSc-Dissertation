
# MSc Dissertation — Explainable Ensemble ML Pipeline

This repository contains the full machine learning pipeline developed for my MSc dissertation. The aim of my dissertation was to explore and apply explainable AI (XAI) frameworks to a biomedical dataset in order to investigate and gain meaningful insights into the biomarkers associated with gram positive and negative infections, their relative importance and the potential threshold values for clinical interpretation.

This demonstrates a hybrid engineering + research approach combining:

- Robust cross‑validated model training  
- Ensemble prediction  
- Multiple explainability methods (SHAP, LIME, IG, permutation importance)  
- Feature stability analysis  
- Surrogate decision‑tree threshold extraction  
- Graph‑theoretic feature correlation networks  
- Critical‑case selection for local interpretability  

The project is structured as a modular, production‑minded codebase.

## Structure
MSc-Dissertation/
│
├── src/
│   ├── data/                # data loading & preprocessing 
│   ├── models/              # training, evaluation, ensemble
│   ├── explainability/      # SHAP, LIME, IG, permutation importance
│   ├── analysis/            # feature stability, graph theory, critical cases
│   └── utils/               # plotting, IO helpers
│
├── scripts/                 # runnable pipeline scripts
├── notebooks/               # clean analysis notebooks
├── data/                    # raw + processed data
└── README.md


---

## Key Features

### **1. Cross‑validated training**
- 5‑fold stratified CV  
- Oversampling applied only to training folds  
- Fold models saved as `model_1.keras` … `model_5.keras`

### **2. Ensemble model**
- Averages predictions from all fold models  
- Provides stable, low‑variance outputs  
- Wrapped in a clean `EnsembleWrapper` class

### **3. Explainability**
- **SHAP** (global + local)
- **LIME** (local)
- **Permutation importance**
- **Integrated Gradients**
- **Surrogate decision tree** for threshold extraction

### **4. Stability analysis**
- Fold‑wise SHAP rank correlations  
- Kruskal–Wallis tests  
- Consensus ranking  
- Top‑k frequency across folds  

### **5. Graph‑theoretic analysis**
- Spearman correlation network  
- Ego‑graphs for key features  
- 3D Plotly visualisation  

### **6. Critical case selection**
- Strong positives  
- Misclassifications  
- Borderline cases  
- Nearest‑neighbour contrastive pairs  


## License

This project is for academic and demonstration purposes.
