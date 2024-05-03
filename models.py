import torch
import torch.nn as nn
import pandas as pd




temp_data=pd.read_csv('Weather Data in India from 1901 to 2017.csv')

temp_data=temp_data.drop(columns=['Unnamed: 0'])
month_sequence = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
mtoi={s:i for i,s in enumerate(month_sequence)}
hidden_size = 64
class TempModel(nn.Module):
    def __init__(self, hidden):
        super(TempModel, self).__init__()
        self.hidden = hidden
        self.blocks = nn.ModuleList()
        for _ in range(12):
            block = nn.Sequential(
                nn.Linear(in_features=1, out_features=hidden),
                nn.Linear(in_features=hidden, out_features=hidden),
                nn.Linear(in_features=hidden, out_features=hidden),
                nn.Linear(in_features=hidden, out_features=hidden),
                nn.ReLU()
            )
            self.blocks.append(block)
        self.output_layer = nn.Linear(in_features=hidden, out_features=1)

    def forward(self, year, month):
        x = self.blocks[month](year)
        x = self.output_layer(x)
        return x

rain_data=pd.read_csv('rainfall in india 1901-2015.csv')
data=rain_data.fillna(1)
states=sorted(set(rain_data['SUBDIVISION']))
months_list = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
mtoi={m:i for i,m in enumerate(months_list)}


class RainFallModel(nn.Module):
    def __init__(self, hidden):
        super(RainFallModel, self).__init__()
        self.state_dist = nn.ModuleDict()
        self.output = nn.Linear(in_features=hidden, out_features=1)

        for state in states:
            self.state_dist[state] = nn.ModuleDict()
            blocks_list = nn.ModuleList()
            for _ in range(12): 
                  block = nn.Sequential(
                        nn.Linear(in_features=1, out_features=hidden),
                        nn.Linear(in_features=hidden, out_features=hidden),
                        nn.Linear(in_features=hidden, out_features=hidden),
                        nn.Linear(in_features=hidden, out_features=hidden),
                        nn.ReLU()
                    )
                  blocks_list.append(block)
            self.state_dist[state] = blocks_list

    def forward(self, year, month, state):
        blocks = self.state_dist[state]
        block = blocks[month]
        x = block(year)
        output = self.output(x)
        return output


pred_data=pd.read_csv('new_main_data.csv')

crops=sorted(set(pred_data['crop']))

ctoi={c:i for i,c in enumerate(crops)}

ctoi

hidden=14
class predictor(nn.Module):
  def __init__(self):
    super().__init__()
    self.modules_dict=nn.ModuleDict()
    self.embedding=nn.Embedding(len(crops),6)

    for i,crop in enumerate(crops):
      self.modules_dict[crop]=nn.Sequential(
          nn.Linear(in_features=6,out_features=hidden),
          nn.Linear(in_features=hidden,out_features=hidden),
          nn.Linear(in_features=hidden,out_features=hidden),
          nn.Linear(in_features=hidden,out_features=hidden),
          nn.ReLU()
      )
    self.out=nn.Linear(in_features=hidden,out_features=6)


  def forward(self,cropidx,crop_name):
    block=self.modules_dict[crop_name]
    x=block(self.embedding(cropidx))
    return self.out(x)



out=pd.read_csv('fertilizer_expanded.csv')

fertilizer=sorted(set(out['fertilizer']))

class fertilizerModel(nn.Module):
  def __init__(self,hidden):
    super().__init__()
    self.block=nn.Sequential(
        nn.Linear(in_features=3,out_features=hidden),
        nn.Linear(in_features=hidden,out_features=hidden),
        nn.Linear(in_features=hidden,out_features=hidden),
        nn.Linear(in_features=hidden,out_features=len(fertilizer))
    )

  def forward(self,x):
    return self.block(x)