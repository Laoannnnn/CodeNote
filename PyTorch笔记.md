## PyTorch

1. **torch.nn.Embedding(num_embeddings, embedding_dim)**

   这个语句是创建一个词嵌入模型，`num_embeddings`代表一共有多少个词，`embedding_dim`代表你想要为每个词创建一个多少维的向量来表示它。

2. `model.train()`：如果model是train的状态，intermediate varaible和computation graph会被保留，这些将来都会在backprop的时候用来计算gradient。因此，速度会比eval慢。

   `model.eval()`：如果model是eval的状态，intermediate variable和computation graph不会被保留。因此速度会比train快。