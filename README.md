# AI 100 – Final Project
**AI System Building with GenAI as a Cognitive Partner**  
 
---
 
## Project Overview
This project uses a simple CNN trained on CIFAR-10 for image classification.  
The goal is to intentionally introduce bugs into the system, reflect on what causes them, and use GenAI as a Socratic tutor to deepen understanding.
 
---
 
## Files
```
├── train.py               # Base model (no bugs) — SimpleCNN on CIFAR-10
├── bug_cases_FINAL.xlsx   # 10 bug cases with reflections and GenAI guidance
├── final_report.pdf       # Written report on learnings
└── bugs/
    ├── bug01_wrong_flatten.py     # Wrong flatten size in fc1
    ├── bug02_no_zero_grad.py      # Missing optimizer.zero_grad()
    ├── bug03_wrong_loss.py        # MSELoss instead of CrossEntropyLoss
    ├── bug04_high_lr.py           # Learning rate = 10.0
    ├── bug05_wrong_channels.py    # Wrong input channels (1 instead of 3)
    ├── bug06_no_eval.py           # Missing model.eval() during testing
    ├── bug07_no_shuffle.py        # shuffle=False on training data
    ├── bug08_zero_epochs.py       # EPOCHS = 0
    ├── bug09_zero_batch.py        # BATCH_SIZE = 0
    └── bug10_wrong_classes.py     # 2 output classes instead of 10
```
 
---
 
## Model
- **Dataset:** CIFAR-10 (10 classes, 60,000 images)
- **Architecture:** SimpleCNN — 2 conv layers + 2 fully connected layers
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 5
---
 
## LLM Used
Claude — claude-sonnet-4-20250514
 
