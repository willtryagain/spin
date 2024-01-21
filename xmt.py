import torch

emb = torch.nn.Embedding(4, 1)
index = torch.LongTensor([2])
emb(index)[0][0]
a = torch.randn(3, 4).to("cuda:1")

a + emb(index)[0][0]
