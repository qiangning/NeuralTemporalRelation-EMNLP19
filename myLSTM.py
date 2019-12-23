import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class lstm_siam(nn.Module):  # add categorical emb to two places instead of one
    def __init__(self, params,emb_cache,bigramGetter,granularity=0.05, common_sense_emb_dim=64,bidirectional=False,lowerCase=False):
        super(lstm_siam, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
        self.nn_hidden_dim = params.get('nn_hidden_dim',32)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim',4)
        self.batch_size = params.get('batch_size',1)
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.common_sense_emb = nn.Embedding(int(1.0/self.granularity)*self.bigramStats_dim,self.common_sense_emb_dim)
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim // 2,\
                                num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim,\
                                num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(2*self.lstm_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.nn_hidden_dim)
        self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.output_dim)
        self.init_hidden()
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()
    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
                           torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))

    def forward(self, temprel):
        self.init_hidden()

        # common sense embeddings
        bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
        common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor(
            [min(int(1.0 / self.granularity) - 1, int(bigramstats[0][0] / self.granularity))])).view(1, -1)
        for i in range(1, self.bigramStats_dim):
            tmp = self.common_sense_emb(torch.cuda.LongTensor([(i - 1) * int(1.0 / self.granularity) + min(
                int(1.0 / self.granularity) - 1, int(bigramstats[0][i] / self.granularity))])).view(1, -1)
            common_sense_emb = torch.cat((common_sense_emb, tmp), 1)

        if not self.lowerCase:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
        else:
            embeds = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
        embeds = embeds.view(temprel.length,self.batch_size,-1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
        lstm_out = lstm_out[temprel.event_ix][:][:]
        h_nn = F.relu(self.h_lstm2h_nn(torch.cat((lstm_out.view(1,-1),common_sense_emb),1)))
        output = self.h_nn2o(torch.cat((h_nn,common_sense_emb),1))
        return output
