# Synthetic Languages Analysis with Large Language Models

## Project Overview

This project investigates how Large Language Models (LLMs) handle synthetic languages - artificially constructed languages with specific linguistic properties. The research explores the boundaries of language understanding in transformer-based models by testing their ability to learn and generate text in constructed languages with varying grammatical rules and structures.

## Objectives

1. **Evaluate LLM Performance**: Assess how well modern language models can adapt to synthetic languages with different linguistic properties
2. **Linguistic Structure Analysis**: Investigate which types of linguistic structures are easier or harder for LLMs to learn
3. **Transfer Learning**: Explore how knowledge from natural languages transfers to synthetic language understanding

## Methodology

### Synthetic Language Generation

The project implements several types of synthetic languages:

- **Hop Languages**: Languages with word/token hopping rules
- **Reverse Languages**: Languages with various reversal patterns
- **Shuffle Languages**: Languages with deterministic and non-deterministic shuffling rules

### Implementation Details

- **Data Generation**: Custom Python scripts to generate synthetic language corpora
- **Model Training**: Fine-tuning pre-trained transformers on synthetic language data
- **Evaluation**: Comprehensive metrics to assess language understanding and generation

## Project Structure

```
01-nlp-synthetic-languages/
├── code/
│   └── prepare_mission_impossible_data.py  # Data preparation script
├── data/
│   ├── synthetic_languages/                 # Generated language samples
│   │   ├── hop/                            # Hopping pattern languages
│   │   ├── reverse/                        # Reversal pattern languages
│   │   └── shuffle/                        # Shuffling pattern languages
│   └── tagged_sentences.json               # Processed sentence data
├── docs/
│   ├── 2401.06416v2.pdf                   # Research paper reference
│   ├── plan_experience_impossible_languages_fr.md
│   └── README.md
└── README.md
```

## Key Findings

1. **Pattern Recognition**: LLMs show varying abilities to recognize and replicate different synthetic patterns
2. **Complexity Threshold**: There appears to be a complexity threshold beyond which model performance degrades significantly
3. **Transfer Learning**: Some linguistic patterns learned from natural languages help in synthetic language understanding

## Technologies Used

- **Python 3.8+**
- **PyTorch**
- **Transformers (Hugging Face)**
- **NumPy & Pandas**
- **Custom synthetic language generators**

## Running the Code

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Languages**:
   ```bash
   python code/prepare_mission_impossible_data.py
   ```

3. **Explore Generated Data**:
   The synthetic languages are stored in JSON format in the `data/synthetic_languages/` directory.

## Future Work

- Expand to more complex linguistic structures
- Test on larger language models
- Investigate multilingual synthetic language learning
- Develop standardized benchmarks for synthetic language evaluation

## References

- Research paper: [2401.06416v2.pdf](docs/2401.06416v2.pdf)
- Related work on impossible languages and LLM limitations

## Author

Master's in Data Science Student

---

*This project is part of a Master's degree program in Data Science, focusing on Natural Language Processing and the theoretical limits of language models.*