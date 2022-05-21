import torch
# torch.set_printoptions(profile='full')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from imageFeature_torch import ImageFeature
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
import json

class WordDataset(Dataset):
  def __init__(self, data_file, data_dir, with_image=False):
    super(WordDataset, self).__init__()
    self.data_dir = data_dir
    self.with_image = with_image
    self.data = []
    with open(data_dir + data_file, 'r', encoding='utf-8') as f:
      for line in f.readlines():
        self.data.append(json.loads(line[:-1]))

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    data_item = self.data[idx]
    text = data_item['s']
    o = data_item['o']
    if self.with_image:
      return text, o, self.data_dir + data_item['img_file']
    else:
      return text, o
      
class WordModel(nn.Module):
  def __init__(self, encoder_hidden_dim, device, maxlength=30, 
              with_image=False, image_encode_dim=512, image_token_num=2):
    super(WordModel, self).__init__()
    
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    self.bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    self.bert_model.eval()
    
    for name, parameter in self.bert_model.named_parameters():
      parameter.requires_grad = False
    self.tokenizer_embeddings = self.bert_model.get_input_embeddings()
    self.device = device
    self.max_length = maxlength

    self.with_image = with_image
    if with_image:
      self.imageProcessor = ImageFeature(self.device, 'VGG16')
      self.image_token_num = image_token_num
      self.image_projection = nn.Linear(image_encode_dim, encoder_hidden_dim * self.image_token_num)

  def forward(self, text, o, image=None):
    bs = len(text)
    if self.with_image:
      image_features = self.imageProcessor.get_batch_feature(image)
      # image_features = torch.zeros((bs, 512)).to(self.device)
      image_hidden = self.image_projection(image_features)
      # print(image_hidden)
      image_token = image_hidden.view(bs, self.image_token_num, -1)

      bert_token = self.tokenizer(list(text), 
            max_length=self.max_length-self.image_token_num, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True).to(self.device)
      input_ids = bert_token['input_ids']
      bert_emb = self.tokenizer_embeddings(input_ids)
      attention_mask = bert_token['attention_mask']
      bert_emb = torch.cat([image_token, bert_emb], dim=1)
      attention_mask = torch.cat([torch.ones(bs, self.image_token_num).to(self.device), attention_mask], dim=1)
      mask_pos = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=True)[1]+self.image_token_num
      
    else:
      bert_token = self.tokenizer(list(text), 
            max_length=self.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True).to(self.device)
      # print(bert_token)
      input_ids = bert_token['input_ids']
      bert_emb = self.tokenizer_embeddings(input_ids)
      # print(bert_emb)
      attention_mask = bert_token['attention_mask']
      mask_pos = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=True)[1]
      # print(mask_pos)
    output = self.bert_model(inputs_embeds=bert_emb, attention_mask=attention_mask)
    # print(output)
    logits = output[0]
    re = []
    for i in range(bs):
      re.append(logits[i][mask_pos[i]])
    labels = self.tokenizer(list(o), add_special_tokens=False, return_tensors='pt').input_ids.squeeze(1)
    return torch.stack(re, dim=0).to(self.device), labels.to(self.device)

class AverageMeter:  # 为了tqdm实时显示loss和acc
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_model_dir", type=str, default="./models")
parser.add_argument("--save_model_name", type=str, default="")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--train_file", type=str, default="train_pattern_singleword.jsonl")
parser.add_argument("--valid_file", type=str, default="valid_pattern_singleword.jsonl")
parser.add_argument("--test_file", type=str, default="test_pattern_singleword-1.jsonl")
parser.add_argument("--do_train", action="store_true", default=False)
parser.add_argument("--do_eval", action="store_true", default=False)
parser.add_argument("--filter", action="store_true", default=False)
parser.add_argument("--with_image", action="store_true", default=False)
parser.add_argument("--log_dir", type=str, default="runs")
parser.add_argument("--dataset_dir", type=str, default="../benchmarks/FB15K_bert/")
parser.add_argument("--all_name_file", type=str, default="all_entity_name.txt")

args = parser.parse_args()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def test():
  with open(args.dataset_dir + 's2o_list.json', 'r') as f:
    s2o_list = json.loads(f.read())
  test_data = WordDataset(args.test_file, args.dataset_dir, args.with_image)
  test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
  baseModel = WordModel(768, device, with_image=args.with_image)
  if len(args.save_model_name) != 0:
    baseModel.load_state_dict(torch.load(args.save_model_dir + '/' + args.save_model_name))
  baseModel = baseModel.to(device)
  baseModel.eval()

  tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
  mrank = torch.empty((0,), dtype=torch.float32)
  mrr = torch.empty((0,), dtype=torch.float32)
  hits = [0, 0, 0]
  test_num = 0
  with torch.no_grad():
    for _, data in enumerate(tk):
      logits, labels = baseModel(*data)
      logits = logits.cpu()
      labels = labels.cpu()
      if args.filter:
        texts = data[0]
        for i in range(len(texts)):
          if len(s2o_list[texts[i]]) > 1:
            for index_ in s2o_list[texts[i]]:
              if index_ != labels[i]:
                logits[i, index_] = 0
      sorted_logits = torch.sort(-logits, dim=-1)[1]
      labels = labels.unsqueeze(1).expand(sorted_logits.shape)
      # print(labels)
      rank = torch.nonzero(sorted_logits == labels, as_tuple=True)[1]+1
      mrank = torch.cat([mrank, rank], dim=0)
      mrr = torch.cat([mrr, 1/rank], dim=0)
      
      for i, k in enumerate([1, 5, 10]):
        hits_k = torch.sum(torch.any(labels[:, :k] == sorted_logits[:, :k], dim=-1)).item()
        hits[i] += hits_k
      test_num += labels.shape[0]
  print('mrr: {:.4f}\t mrank: {:.4f}\t hits@1: {:.4f}\t hits@5: {:.4f}\t hits@10: {:.4f}'
              .format(mrr.mean().item(), mrank.mean().item(), hits[0]/test_num, hits[1]/test_num, hits[1]/test_num))

def valid(loader, baseModel):
  cnt = 0
  tk = tqdm(loader, total=len(loader), position=0, leave=True)
  with torch.no_grad():
    for _, data in enumerate(tk):
      logits, labels = baseModel(*data)
      preds = logits.argmax(dim=-1)
      cnt += torch.sum(preds == labels).item()
  return cnt

def train():
  print('learning rate: {}\nbatch_size: {}\n'.format(args.lr, args.batch_size))
  if args.with_image:
    print('with_image')
  train_data = WordDataset(args.train_file, args.dataset_dir, args.with_image)
  valid_data = WordDataset(args.valid_file, args.dataset_dir, args.with_image)
  # vocab_size = 6783

  baseModel = WordModel(768, device, with_image=args.with_image).to(device)
  train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
  # optimizer = AdamW(baseModel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  optimizer = AdamW(baseModel.parameters(), lr=args.lr)
  losses = AverageMeter()

  best_cnt = 0
  for epoch in range(args.epochs):
    loss_fn = nn.CrossEntropyLoss()
    tk = tqdm(train_loader)
    for data in tk:
      logits, labels = baseModel(*data)
      loss = loss_fn(logits, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses.update(loss.item(), len(data))
      tk.set_postfix(loss=losses.avg)

    # baseModel.eval()
    cnt = valid(valid_loader, baseModel)
    eval_acc = 1.0*cnt/valid_data.__len__()
    print('epoch {}: eval_acc: {:.4f}\n'.format(epoch, eval_acc))

    if best_cnt < cnt:
      best_cnt = cnt
      if args.with_image:
        model_name = '{}/mmkg-lr_{}-acc_{:.3f}.pth'.format(args.save_model_dir, args.lr, eval_acc)
      else:
        model_name = '{}/predict_word_lr_{}-acc_{:.3f}.pth'.format(args.save_model_dir, args.lr, eval_acc)
      torch.save(baseModel.state_dict(), model_name)

if args.do_train:
  train()
else:
  test()
