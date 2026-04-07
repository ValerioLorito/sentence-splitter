# BERT Sentence Splitter
This project implements a Sentence Splitting system based on fine-tuning BERT models for two different languages: Italian and English. The system is designed to outperform traditional rule-based baselines (NLTK) and syntactic dependency baselines (spaCy) by successfully tackling anomalies and linguistic "dataset shifts" through Data Augmentation techniques. The key feature which has been introduced in the Data Augmentation process is the Dirty Probability: this setting defines a techinque for which the dataset tokens that preceed \<EOS> occurrencies can be deleted or swapped wih other tokens with an arbitrary probability rate. 

## 📁 Reference Datasets
Training and comprehensive evaluation (on both `dev` and `test` splits) were strictly performed on the official Universal Dependencies (UD) datasets provided as a reference for this project:

* **Italian Datasets:** `UD_Italian-ISDT`, `UD_Italian-MarkIT`, `UD_Italian-ParTUT`, `UD_Italian-VIT`.
* **English Datasets:** `UD_English-EWT`, `UD_English-GUM`, `UD_English-ParTUT`, `UD_English-PUD`.

## ⚙️ 1. Initialization and Environment Setup
To ensure perfect reproducibility of the project, it is highly recommended to use a virtual environment.

**Step 1: Create and activate the virtual environment**
```bash
python -m venv venv
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Step 2: Install dependencies**
The 'requirements.txt' file contains all the necessary libraries (including torch, transformers, nltk, and spacy).

```bash
pip install -r requirements.txt
```
**Step 3: Download spaCy models**
To run the evaluation scripts and compare them with the baselines, you must download the statistical spaCy models for Italian and English:

```bash
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
```

## 🚀 2. Best Performing Models (Best Runs)
During the training and validation phases, multiple experiments were conducted. The final models selected for the official Test Set evaluation are the following:

**🇮🇹 Italian Model**
This model was trained to handle the Italian language, which generally features a cleaner structure, thus requiring a lighter preventive Data Augmentation. The best training setting results the following:

Base Model: dbmdz/bert-base-italian-xxl-cased  
Learning Rate: [INSERT LR]  
Epochs: [INSERT EPOCHS]  
Weight Decay: [INSERT WD]  
Dirty Prob (Train): [INSERT PROBABILITY]  

**🇬🇧 English Model**
This model was trained for English, applying a more aggressive Data Augmentation to counter extremely chaotic datasets (e.g., EWT and GUM) full of quotes, brackets, and unpunctuated headers.  

Base Model: bert-base-cased  
Learning Rate: [INSERT LR]  
Epochs: [INSERT EPOCHS]  
Weight Decay: [INSERT WD]  
Dirty Prob (Train): [INSERT PROBABILITY]  

## 📊 3. Evaluation and Results (Test Sets)
The models were rigorously evaluated on the official Test Sets in a direct comparison against NLTK (Punkt) and spaCy. The prediction extraction process was perfectly fair: BERT worked on context-aware chunks, while NLTK and spaCy processed the raw text to prevent altering their internal space-based rules.

A brief comparison of the best fine-tuned BERT models with the NLTK and Spacy models is given below:

**Italian Results**
Dataset	Metric | NLTK (Punkt) |	spaCy (it_core_news) | Our Model  
_ISDT_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  
_MARKIT_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  
_PARTUT_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  
_VIT_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  

**English Results**
Dataset	Metric | NLTK (Punkt) | spaCy (en_core_web) | Our Model  
_EWT_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  
_GUM_	  
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  
_PARTUT_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]  
_PUD_	 
F1-Score	[INSERT]	[INSERT]	[INSERT]  
Precision	[INSERT]	[INSERT]	[INSERT]  
Recall	  [INSERT]	[INSERT]  [INSERT]  
Accuracy	[INSERT]	[INSERT]	[INSERT]   
