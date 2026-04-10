# IPL Player Performance Analytics — Datathon Project

## Team Details
| Name | Roll No | Role |
|------|---------|------|
| Dhruv | 57 | Data Wrangler & Model Developer |
| Onkar | 60 | EDA & Visualizer |
| Parth | 36 | Reporter & Dashboard Creator |

## Problem Statement
To analyze IPL (Indian Premier League) historical match and ball-by-ball data to:
1. Identify key factors that influence match outcomes
2. Predict match-winning probability using ML
3. Cluster player performance profiles

## Key Findings
- Teams winning the toss and choosing to field won ~52% of matches
- First innings score >170 runs correlates strongly with match wins
- Random Forest model achieved ~72% accuracy (AUC: 0.78) for win prediction
- 4 distinct player profiles identified: Power Hitters, Anchors, Consistent Scorers, Finishers
- Virat Kohli and David Warner dominate the "Consistent Scorer" cluster

## 📁 Repository Structure

```
ipl-datathon/
├── 📓 notebooks/       → Jupyter notebooks for each phase
├── 📊 data/            → Raw and processed CSV files
├── 📤 outputs/         → Charts (PNG) and model metrics (CSV)
├── 📊 dashboard/       → Power BI .pbix file
├── 📝 report/          → Final report
└── 🐍 src/             → Standalone Python model script
```

## How to Reproduce
1. Clone this repo
2. pip install -r requirements.txt
3. Run notebooks in order: preprocessing → EDA → modeling
4. Open dashboard/ipl_dashboard.pbix in Power BI Desktop

## Dataset
- **Source:** Kaggle — IPL Complete Dataset 2008–2020
- **Files:** matches.csv (756 rows), deliveries.csv (179,078 rows)
- **Link:** https://www.kaggle.com/datasets/yash9439/ipl-dataset

## Tentative Methodology
- Tools: Python (pandas, sklearn, matplotlib, seaborn), Power BI
- ML: Random Forest Classifier for win prediction, K-Means for player clustering.
