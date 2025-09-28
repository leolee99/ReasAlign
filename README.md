# ReasAlign

This is the repository for the paper ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack.

In this repository, we provide our fine-tuned models (including the defensive model and the logic judge model), the reproduction code, and the result files.

## How to start

### Environment Setup

Download Dependencies:
```
Python=3.10
pip install -r requirements.txt
```

Then, you should set up your ```OPENAI_API_KEY``` for evaluating utility and ASR.  Please run:

```
export OPENAI_API_KEY=your_key
```

### Evaluations

If you want to evaluate on CyberSecEval2, run:
```bash scripts/cyse.sh```.

If you want to evaluate on AlpacaEval, run:
```bash scripts/alpaca.sh```

If you want to evaluate on SEP utility benchmark, run:
```bash scripts/sep_utility.sh```

If you want to evaluate on SEP security benchmark, run:
```bash scripts/sep_security.sh```

The generated results file will be saved in ```results/```. 

**Note:** We have provided the results file reported in the paper here, which you can directly refer to.
