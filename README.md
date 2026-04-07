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
Learning Rate: [2e-05]      
Epochs: [10]        
Weight Decay: [0.1]     
Dirty Prob (Train): [0.1]       

**🇬🇧 English Model**
This model was trained for English, applying a more aggressive Data Augmentation to counter extremely chaotic datasets (e.g., EWT and GUM) full of quotes, brackets, and unpunctuated headers.  

Base Model: bert-base-cased  
Learning Rate: [3e-05]  
Epochs: [10]  
Weight Decay: [0.1]  
Dirty Prob (Train): [0.15]  

## 📊 3. Evaluation and Results (Test Sets)
The models were rigorously evaluated on the official Test Sets in a direct comparison against NLTK (Punkt) and spaCy. The prediction extraction process was perfectly fair: BERT worked on context-aware chunks, while NLTK and spaCy processed the raw text to prevent altering their internal space-based rules.

A brief comparison of the best fine-tuned BERT models with the NLTK and Spacy models is given below:

**Italian Results**
Dataset	Metric | NLTK (Punkt) |	spaCy (it_core_news) | Our Model        
_ISDT_	    
F1-Score	|   [0.9539]	|   [0.9866]    |	[0.9959]    
Precision	|   [0.9640]    |	[0.9835]    |	[0.9938]    
Recall	    |   [0.9440]    |	[0.9896]    |    [0.9979]    
Accuracy	|   [0.9959]    |	[0.9988]    |	[0.9996]    
_MARKIT_	
F1-Score	|   [0.9171]    |	[0.9838]    |	[1.0000]    
Precision	|   [0.9799]    |	[0.9824]    |	[1.0000]    
Recall	    |   [0.8618]    |	[0.9853]    |    [1.0000]    
Accuracy	|   [0.9950]    |	[0.9990]    |	[1.0000]    
_PARTUT_	    
F1-Score    |	[0.9934]    |	[0.9934]    |	[1.0000]    
Precision   |	[1.0000]    |	[1.0000]    |	[1.0000]    
Recall  	|   [0.9869]    |	[0.9869]    |    [1.0000]    
Accuracy	|   [0.9995]    |	[0.9995]    |	[1.0000]    
_VIT_	    
F1-Score	|   [0.9584]    |	[0.9189]    |	[0.9769]    
Precision	|   [0.9795]    |	[0.8760]    |	[0.9627]    
Recall	    |   [0.9381]    |	[0.9663]    |    [0.9916]    
Accuracy	|   [0.9968]    |	[0.9933]    |	[0.9982]    

**English Results**
Dataset	Metric | NLTK (Punkt) | spaCy (en_core_web) | Our Model     
_EWT_	
F1-Score    |   [0.8028]    |	[0.8194]    |	[0.9282]    
Precision	|   [0.9833]    |	[0.9528]    |	[0.9483]    
Recall	    |   [0.6784]    |	[0.7188]    |	[0.9090]    
Accuracy	|   [0.9774]    |	[0.9785]    |	[0.9904]    
_GUM_	    
F1-Score	|   [0.9186]    |	[0.9229]    |	[0.9733]    
Precision	|   [0.9711]    |	[0.9574]    |	[0.9753]    
Recall	    |   [0.8716]    |  	[0.8907]    |	[0.9713]     
Accuracy	|   [0.9929]    |	[0.9931]    |	[0.9975]    
_PARTUT_	    
F1-Score	|   [0.9934]    |	[0.9869]    |	[0.9935]    
Precision	|   [1.0000]    |	[0.9869]    |	[0.9935]    
Recall	    |   [0.9869]    |	[0.9869]    |	[0.9935]    
Accuracy	|   [0.9995]    |	[0.9989]    |	[0.9995]    
_PUD_	
F1-Score	|   [0.9910]    |	[0.9965]    |	[0.9891]    
Precision	|   [0.9900]    |	[0.9950]    |	[0.9813]    
Recall	    |   [0.9920]    |	[0.9980]    |    [0.9970]    
Accuracy	|   [0.9992]    |	[0.9997]    |	[0.9991]    

## 💻 4. Running the Code (CLI Commands)
Here are the commands to reproduce the different phases of the project from your terminal. 

**Dataset Analysis**
To extract dynamic metrics and `<EOS>` statistics from the datasets:
```bash
python dataset_analysis/analyze_datasets.py
```

**Final Test Set Evaluation (With Pre-trained Models)**
If you do not want to train the models from scratch, you can use the best performing models uploaded to the Hugging Face Hub. The evaluation script will automatically download the weights and run the comparison against NLTK and spaCy.

To test the **Italian** model:
```bash
python src/evaluate_all.py --run valeriolorito/bert-italian-sentence-splitter --lang italian
```

To test the **English** model:
```bash
python src/evaluate_all.py --run valeriolorito/bert-english-sentence-splitter --lang english
```

If you want to train, evaluate and see the test results, follow the guidelines below:

**Model Training(optional)**
If you want to train a new BERT model from scratch. Replace 'LANGUAGE' with either 'italian' or 'english':
```bash
python src/train.py --lang LANGUAGE
```

**Dev Set Evaluation & Error Analysis(optional)**
To evaluate a specific run on the Dev Sets, generating comparison confusion matrices and detailed error logs. Replace 'RUN_NAME' with the exact name of your run folder (e.g., 'run_english_2026'...) and 'LANGUAGE' with 'italian' or 'english':
```bash
python src/error_analysis.py --run RUN_NAME --lang LANGUAGE
```

**Final Test Set Evaluation(optional)**
To evaluate a trained model on the official Test Sets, generating the final JSON metrics and the comparison bar charts:
```bash
python src/evaluate_all.py --run RUN_NAME --lang LANGUAGE
```

Enjoy!
