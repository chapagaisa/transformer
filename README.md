# transformer

### Advancing Minority Stress Detection with Transformers

![Image](https://github.com/user-attachments/assets/4d109072-b786-4be7-8e43-b449d4b467eb)
Fig. 1: Encoder-Based (Left) and Decoder-Based (Right) Transformer Architectures

![Image](https://github.com/user-attachments/assets/e4796a33-c63f-4b6a-af58-120feb342bf0)
Fig. 2: BERT-GCN Network Architecture.

![Image](https://github.com/user-attachments/assets/06ed965b-e326-4da2-9193-76e301be773f)
Fig. 3: BERT-CNN network architecture

![Image](https://github.com/user-attachments/assets/f790a5f4-8979-4e6b-ade5-d5e3d2662b59)
Fig. 4: BERT-BiGRU network architecture.

### Dependencies
We used a Linux server equipped with an NVIDIA GPU (Please see Experiments section of Paper for details), which has Python Version of 3.10.12, with following required packages to train our models: <br>
1. torch==2.0.1
2. transformer==4.30.2
3. torchtext==0.6
4. numpy==1.22.4 
5. pandas==1.5.3 


### Instructions
Step1: Install dependencies using terminal **pip install -r requirements.txt**. <br>
Step2: Run **python <filename>.py** <br>

*NOTE: Due to ethical concerns, datasets are not made public. However, you can test the model with any publicly available datasets (such as Movie Reviews, Book Reviews, etc.).


