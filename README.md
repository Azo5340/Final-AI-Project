# AI 100 – Final Project
*AI System Building with GenAI as a Cognitive Partner*
 

 
## Project Overview
This project uses a simple CNN trained on CIFAR-10 for image classification.  
The goal is to intentionally introduce bugs into the system, reflect on what causes them, and use GenAI as a Socratic tutor to deepen understanding.
 

 
## Files
```
├── bugs:
    ├── bug01_wrong_flatten.py     
    ├── bug02_no_zero_grad.py      
    ├── bug03_wrong_loss.py        
    ├── bug04_high_lr.py           
    ├── bug05_wrong_channels.py    
    ├── bug06_no_eval.py           
    ├── bug07_no_shuffle.py        
    ├── bug08_zero_epochs.py       
    ├── bug09_zero_batch.py        
    └── bug10_wrong_classes.py  
├── train.py                  # Base model: SimpleCNN on CIFAR-10
├── bug_cases(Bug Cases).csv  # Excel file of the 10 bug cases with reflections and GenAI guidance
├── final_report.pdf          # PDF report 
└── bugs:    
```
 

 
## Model
- **Dataset:** CIFAR-10 (10 classes, 60,000 images)
- **Architecture:** SimpleCNN — 2 conv layers + 2 fully connected layers
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 5

 
## LLM Used
Claude — claude-sonnet-4-20250514

## Author 
Adam Ouareth | Penn State University Park | Spring 2026
 
