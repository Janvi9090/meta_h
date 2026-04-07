# Medication Dosing & Toxicity Control (OpenEnv)

## Overview
This environment simulates drug dosing in a patient where the agent must maintain drug concentration within a therapeutic window.

## Objective
Keep drug concentration between 10–50 while avoiding toxicity (>70).

## Tasks
- Easy: Fixed metabolism
- Medium: Variable metabolism
- Hard: Higher metabolism complexity

## Reward
+1 → within safe range  
-0.5 → underdose  
-1 → risky high  
-2 → toxic  

## How to Run
```bash
python inference.py


---

# 🚀 DONE — What You Do Next

1. Copy all files into a folder  
2. Test locally:
   ```bash
   python inference.py