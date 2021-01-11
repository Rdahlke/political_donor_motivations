#####
# title: 3. Classification
# author: Ross Dahlke
# date: 1/10/2020
#####

# import packages
import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, InputFeatures, AdamW, BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import functional as F

# read in pre-coded data
coded = pd.read_csv("data/posts_coded.csv")

# reformat to long
coded_long = pd.melt(coded, id_vars = ["text"]).dropna(thresh = 3).reset_index()

# create a key for the labels
variable_key = pd.DataFrame({"variable": coded_long["variable"].unique()})
variable_key["label"] = variable_key.index
coded_long = coded_long.merge(variable_key)

# split into train and test datasets
trn_idx, test_idx = train_test_split(np.arange(len(coded_long)), test_size = .5, random_state = 1)

# load in the large BERT model
model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

# loving that cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device) # without this there is no error, but it runs in CPU (instead of GPU).
model.eval() # declaring to the system that we're only doing 'forward' calculations

# model set up stuff
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": .01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr = 1e-5)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# preparing training batch
train_batch = [coded_long["text"][i] for i in trn_idx]
train_encoding = tokenizer(train_batch, return_tensors='pt', padding=True, truncation=True, max_length = 50)
train_input_ids = train_encoding['input_ids'].to(device)
train_input_ids = train_input_ids.type(dtype = torch.long)
train_attention_mask = train_encoding['attention_mask'].to(device).float()
train_labels = torch.tensor([coded_long["label"][i] for i in trn_idx])
train_labels = train_labels.type(torch.long)
train_labels = train_labels.to(device)

# preparing test batch
test_batch = [coded_long["text"][i] for i in test_idx]
test_encoding = tokenizer(test_batch, return_tensors='pt', padding=True, truncation=True, max_length = 50)
test_input_ids = test_encoding['input_ids'].to(device)
test_input_ids = test_input_ids.type(dtype = torch.long)
test_attention_mask = test_encoding['attention_mask'].to(device).float()
test_labels = torch.tensor([coded_long["label"][i] for i in test_idx])
test_labels = test_labels.type(torch.long)
test_labels = test_labels.to(device)

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])
    return batch

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total # of training epochs
    per_device_train_batch_size=1000,  # batch size per device during training
    per_device_eval_batch_size=1000,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',
    save_total_limit=1,
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    data_collator = dummy_data_collector
)

trainer.train()

torch.cuda.empty_cache()

trainer.evaluate()

print((model(test_input_ids, test_attention_mask, labels = test_labels)))

output = model(test_input_ids, test_attention_mask)

_, prediction = torch.max(output, dim = 1)

torch.max(F.softmax(output[0]), dim = 1)

output.preds

F.softmax(model.predict(test_dataset))

output

_, prediction = torch.max(output, dim=1)

prediction



def get_predictions(model, text_batch, input_ids, attention_mask, labels):
  model = model.eval()
  texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for i in range(len(text_batch)):
      texts = text_batch
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      targets = labels.to(device)
      outputs = model(
        input_ids = input_ids,
        attention_mask = attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_batch,
    test_input_ids,
    test_attention_mask,
    test_labels
)

def get_predictions(model, text_batch, input_ids, attensio):
  model = model.eval()
  texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for i in range(len(test_dataset)):
      texts = test_batch[0]
      input_ids = test_dataset[0][0].to(device)
      attention_mask = test_dataset[0][1].to(device)
      targets = test_dataset[0][2].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs)
      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values
