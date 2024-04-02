# BERT_BiGRU

### Classification of LGBTQ+ Minority Stress with BERT-BiGRU

![BERT_GRU_ARC](https://github.com/chapagaisa/BERT_BiGRU/assets/46834070/abc6a851-d8aa-4634-99c8-2ccef78fb8b2)
Figure1. BERT-BiGRU network architecture.

### Dependencies
We used a Linux server equipped with an NVIDIA GPU (Please see Experiments section of Paper for details), which has Python Version of 3.10.12, with following required packages to train our models: <br>
1. torch==2.0.1
2. transformer==4.30.2
3. torchtext==0.6
4. numpy==1.22.4 
5. pandas==1.5.3 


### Instructions
Step1: Install dependencies using terminal **pip install -r requirements.txt**. <br>
Step2: Run **python roberta_bigru.py** <br>

*NOTE: Due to ethical concerns, datasets are not made public. However, you can test the model with any publicly available datasets (such as Movie Reviews, Book Reviews, etc.).
### Files
1. roberta_bigru.py: Proposed model.
2. Baseline_BiLSTM.ipynb: BiDirectional LSTM with pre-trained (GloVe) word embeddings.
3. Baseline_ML_Classifiers.ipynb: SVM, Naive Bayes, Logistic Regression, Random Forests, AdaBoost, MLP.

