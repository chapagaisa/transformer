# BERT_BiGRU

### Classification of LGBTQ+ Minority Stress on Social Media with a Pre-Trained BERT-BiGRU Model

![BERT_GRU_ARC](https://github.com/chapagaisa/BERT_BiGRU/assets/46834070/abc6a851-d8aa-4634-99c8-2ccef78fb8b2)
Figure1. BERT-BiGRU network architecture.

### Dependencies
The models has been tested in Google Colab which has Python Version of 3.10.12, with following required packages: <br>
1. torch==2.0.1
2. transformer==4.30.2
3. torchtext==0.6
4. numpy==1.22.4 
5. pandas==1.5.3 


### Instructions
Step1: Due to Ethical concerns datasets are not made public. However, you can test the model with any datasets (Moview Review, Book Review, etc.) that is publicly available. <br>
Step2: Install dependencies using terminal "pip install -r requirements.txt". If you wish to use .ipynb file in google colab, dependencies are included in the file. <br>

### Files
1. AblationStudy_BERTONLY.ipynb: Contains BERT and a linear layer for ablation study.
2. AblationStudy_GRUONLY: BiDirectional GRU with custom word embeddings for ablation study.
3. AblationStudy_GRUONLY_Glove: BiDirectional GRU with pre-trained (GloVe) word embeddings for ablation study.
4. BERT_BiGRU: Proposed model.
5. Baseline_BiLSTM: BiDirectional LSTM with pre-trained (GloVe) word embeddings.
6. Baseline_ML_Classifiers: SVM, Naive Bayes, Logistic Regression, Random Forests, AdaBoost, MLP.

