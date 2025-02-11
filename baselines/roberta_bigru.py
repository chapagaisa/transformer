from torchtext.data import Field, TabularDataset, BucketIterator,LabelField
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import torch.nn as nn
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse


parser = argparse.ArgumentParser(description="Minority Stress Analysis Model Arguments")
# Add arguments
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
parser.add_argument("--max_input_length", type=int, default=512, help="Maximum input length")
parser.add_argument("--model_type", type=str, default="roberta-base", choices=["roberta-base", "roberta-large", "bert-base-uncased", "bert-large-uncased"], help="Type of model to use")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the RNN")
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
args = parser.parse_args()

# Retrieve arguments
BATCH_SIZE = args.batch_size
DROPOUT = args.dropout
HIDDEN_DIM = args.hidden_dim
max_input_length = args.max_input_length
model_type = args.model_type
N_LAYERS = args.num_layers
N_EPOCHS = args.epochs

OUTPUT_DIM = 1 
BIDIRECTIONAL = True


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print(torch.version.cuda)
print(torch.__version__)
import sys
print(sys.version)
print(np.__version__)


if model_type.startswith("roberta"):
    tokenizer = RobertaTokenizer.from_pretrained(model_type)
    model_init = RobertaModel.from_pretrained(model_type)
    embedding_dim = model_init.config.hidden_size
elif model_type.startswith("bert"):
    tokenizer = BertTokenizer.from_pretrained(model_type)
    model_init = BertModel.from_pretrained(model_type)
    embedding_dim = model_init.config.hidden_size
else:
    raise ValueError("Invalid model type selected")


init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
print(init_token, eos_token, pad_token, unk_token)

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes[model_type]
print(max_input_length)

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the original dataset
df = pd.read_csv('saha_et_al.csv')
df = df.rename(columns={'label_minority_stress': 'label'})
df = df[['text', 'label']]

empty_cells =  df.isnull().sum()
print(empty_cells)
df.dropna(inplace = True)

empty_cells =  df.isnull().sum()
print(empty_cells)

# Load the data
def load_data():
  train_data,valid_data, test_data = TabularDataset.splits(
      path= '/home/santosh/NLP/BertBiGRU',
      train='train.csv',
      validation='valid.csv',
      test='test.csv',
      format='csv',
      fields=[('text', TEXT), ('label', LABEL)],
      skip_header=True
  )
  return train_data,valid_data, test_data


def create_iterators(train_data, valid_data, test_data):
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
      (train_data, valid_data, test_data),
      batch_size = BATCH_SIZE,
      device = device,
      sort_key=lambda x: len(x.text),
      sort_within_batch = True
      )
  return train_iterator, valid_iterator, test_iterator


class RoBERTa_BiGRU(nn.Module):
    def __init__(self,
                 model_init,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()
        self.model_init = model_init

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.model_init(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        output = self.out(hidden)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch.optim as optim

def get_model_params(model):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    return model, criterion,optimizer

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
def calc_report(model, iterator, criterion, save_roc):
    epoch_loss = 0
    epoch_acc = 0
    y_pred = []
    y_true = []
    test_probs = []  # Store raw probabilities here #roc
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            threshold = 0.5
            output_1 = (predictions > threshold).int()
            y_pred.extend(output_1.tolist())
            y_true.extend(batch.label.tolist())
            test_probs.extend(predictions.detach().cpu().numpy()) #roc
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        rep = classification_report(y_true, y_pred, labels=[1,0], digits=4, output_dict=True)

        if save_roc == True:
            fpr, tpr, thresholds = roc_curve(y_true, test_probs)
            auc_score = roc_auc_score(y_true, test_probs)
            # Save ROC data to a text file
            data = np.column_stack((fpr, tpr, thresholds))
            header = "False Positive Rate (FPR)\tTrue Positive Rate (TPR)\tThresholds"
            np.savetxt(model_type + '_roc_data.txt', data, header=header, delimiter='\t')
            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.4f)' % auc_score)
            plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(model_type + '_roc.png')
            plt.close()

	# Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(model_type + '_confusion_matrix.png')
            plt.close()	
    return rep


def run_epoch(N_EPOCHS):
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []
  best_valid_loss = float('inf')

  for epoch in range(N_EPOCHS):
      start_time = time.time()
      train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
      valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)
      train_accuracies.append(train_acc)
      valid_accuracies.append(valid_acc)
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), model_type +'.pt')
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.4f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.4f}%')
  return train_losses,valid_losses, train_accuracies,valid_accuracies

from sklearn.model_selection import train_test_split
def save_df_to_cvs(df_train,df_test,df_val):
  # Save the dataframes to CSV files
  df_train.to_csv('train.csv', index=False)
  df_test.to_csv('test.csv', index=False)
  df_val.to_csv('valid.csv', index=False)
  time.sleep(5) #make sure the file is available 

train_ratio = 0.65
valid_ratio = 0.15
test_ratio = 0.20

##Running Once
classification_report_dict=[]
i = 0
print("experiment running with random_state = ", i, " ...")
X = df # Contains all columns.
y = df[['label']] # Dataframe of just the column on which to stratify.
# Split original dataframe into train and temp dataframes.
df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                      y,
                                                      stratify=y,
                                                      test_size=(1.0 - train_ratio),
                                                      random_state=i)
# Split the temp dataframe into val and test dataframes.
relative_frac_test = test_ratio / (valid_ratio + test_ratio)
df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                  y_temp,
                                                  stratify=y_temp,
                                                  test_size=relative_frac_test,
                                                  random_state=i)

save_df_to_cvs(df_train,df_test,df_val)

TEXT = Field(batch_first = True,
                use_vocab = False,
                tokenize = tokenize_and_cut,
                preprocessing = tokenizer.convert_tokens_to_ids,
                init_token = init_token_idx,
                eos_token = eos_token_idx,
                pad_token = pad_token_idx,
                unk_token = unk_token_idx)

LABEL = LabelField(dtype = torch.float)
train_data,valid_data, test_data = load_data()
LABEL.build_vocab(train_data)
train_iterator, valid_iterator, test_iterator = create_iterators(train_data, valid_data, test_data)

model = RoBERTa_BiGRU(model_init,
                      HIDDEN_DIM,
                      OUTPUT_DIM,
                      N_LAYERS,
                      BIDIRECTIONAL,
                      DROPOUT)


model, criterion, optimizer = get_model_params(model)
train_losses,valid_losses, train_accuracies,valid_accuracies = run_epoch(N_EPOCHS)

cal_dict = calc_report(model, test_iterator, criterion, True)
print(cal_dict)

#plot graph of train and valid
epochs = range(1, N_EPOCHS + 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(model_type +'loss.png')
plt.close()


plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(model_type+'accuracy.png')
plt.close()

#run only once per experiment(5 differeent random_state of train_test_data_split)

classification_report_dict=[]
for i in range(0,5):
  print("experiment running with random_state = ", i, " ...")
  X = df # Contains all columns.
  y = df[['label']] # Dataframe of just the column on which to stratify.
  # Split original dataframe into train and temp dataframes.
  df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=(1.0 - train_ratio),
                                                        random_state=i)
  # Split the temp dataframe into val and test dataframes.
  relative_frac_test = test_ratio / (valid_ratio + test_ratio)
  df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                    y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    random_state=i)
  save_df_to_cvs(df_train,df_test,df_val)
  TEXT = Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)
  LABEL = LabelField(dtype = torch.float)
  train_data,valid_data, test_data = load_data()
  LABEL.build_vocab(train_data)
  train_iterator, valid_iterator, test_iterator = create_iterators(train_data, valid_data, test_data)
  #Create a new instance for a new training session.
  model = RoBERTa_BiGRU(model_init,
                      HIDDEN_DIM,
                      OUTPUT_DIM,
                      N_LAYERS,
                      BIDIRECTIONAL,
                      DROPOUT)
	
  model, criterion, optimizer = get_model_params(model)
  run_epoch(N_EPOCHS)

  cal_dict = calc_report(model, test_iterator, criterion, False)
  classification_report_dict.append(cal_dict)

def specificCalculation(classification_report_dict):
    Accuracy = []
    precision=[]
    recall=[]
    f1_score=[]
    for i in range(len(classification_report_dict)):
        report=classification_report_dict[i]
        temp=report['accuracy']
        Accuracy.append(temp)

        temp=report['weighted avg']['precision']
        precision.append(temp)

        temp=report['weighted avg']['recall']
        recall.append(temp)

        temp=report['weighted avg']['f1-score']
        f1_score.append(temp)

    print('mean(Accuracy) :',np.mean(Accuracy))
    print('std(Accuracy) :',np.std(Accuracy))
    print('mean np.std(Accuracy) : ',np.round(np.mean(Accuracy),4),"+-",np.round(np.std(Accuracy),4) )
    print('mean np.std(precision) : ', np.round(np.mean(precision),4),"+-",np.round(np.std(precision),4) )
    print('mean np.std(recall) : ', np.round(np.mean(recall),4),"+-",np.round(np.std(recall),4) )
    print('mean np.std(F1-Score) : ', np.round(np.mean(f1_score),4),"+-",np.round(np.std(f1_score),4) )

specificCalculation(classification_report_dict)

