### MISA

[TOC]

* GitHub地址：[https://github.com/declare-lab/MISA](https://github.com/declare-lab/MISA)
* **阅读重点**：
* 怎么通过API调用 `MOSEI` 和 `MOSI` 两个数据集。
* 看 `L_sim` 和 `L_diff` 怎么实现。

#### 1 config.py

1. 对于不同数据集设置了不同的 `batch_size` ：

   > MOSI：64
   >
   > MOSEI：16
   >
   > UR_FUNNY：32

2. 有预训练的 word embedding 文件。



#### 2 create_dataet.py（重点看怎么使用数据集）

##### 2.1 class MOSI

1. 详见笔记 `如何使用MOSI和MOSEI数据集.md`。




#### 3 data_loader.py

##### 3.1 class MSADataset(Dataset)

1. 调用 `create_dataset.py` (2)中的 数据集类 `MOSI/MOSEI/UR_FUNNY`。

##### 3.2 get_loader(config, shuffle=) 

1. 调用 `MSADataset` 类。

2. 定义 `collate_fn()` 方法：

   ```Python
   from torch.nn.utils.rnn import pad_sequence
   
   def collate_fn(batch):
       '''
   	Collate functions assume batch = [Dataset[i] for i in index_set]
   	'''
       # for later use we sort the batch in descending order of length
       # 根据语句长度排序
       batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
   
       # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
       labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
       sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
       visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
       acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
   
       ## BERT-based features input prep
       SENT_LEN = sentences.size(0)
       
       # Create bert indices using tokenizer
       bert_details = []
       for sample in batch:
           text = " ".join(sample[0][3])
           encoded_bert_sent = bert_tokenizer.encode_plus(
               text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)
           bert_details.append(encoded_bert_sent)
   
   
   	# Bert things are batch_first
       bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
       bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
       bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
   
   
       # lengths are useful later in using RNNs
       lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
   
       return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask
   ```

3. 创建 `DataLoader`：

   ```Python
   data_loader = DataLoader(dataset=dataset,
                            batch_size=config.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)
   ```



#### 4 train.py

1. 设置随机种子：

   ```Python
   # Setting random seed
   random_name = str(random())
   random_seed = 336   
   torch.manual_seed(random_seed)
   torch.cuda.manual_seed_all(random_seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   np.random.seed(random_seed)
   ```

2. 调用 `config.py` (1)中的 `get_config(mode=)` 函数。

3. 调用 `data_loader.py` (3)中的 `get_loader(config=, shuffle=)` 函数。

4. 定义 `solver.py` (5)中的 `Solver` 类。

5. 调用 `solver.py` (5.1)中的 `build()` 方法。

6. 调用 `solver.py` (5.2)中的 `train()` 方法。



#### 5 solver.py

##### 5.1 def build()：模型、权重初始化

1. MISA：模型在 `moddels.py` (6) 中定义

   ```Python
   if self.model is None:
       self.model = getattr(models, self.train_config.model)(self.train_config)
   ```
   
2. 初始化权重：

   ```Pyth
   for name, param in self.model.named_parameters():
       if 'weight_hh' in name:
       	nn.init.orthogonal_(param)  # 用一个（半）正定矩阵填充输入张量, 输入张量必须至少有两个维度
   ```

3. 初始化嵌入矩阵的权重，启用cuda，设置优化器：

   ```Python
   # Initialize weight of Embedding matrix with Glove embeddings
   if not self.train_config.use_bert:
       if self.train_config.pretrained_emb is not None:
           self.model.embed.weight.data = self.train_config.pretrained_emb
   	self.model.embed.requires_grad = False
   
   if torch.cuda.is_available() and cuda:
   	self.model.cuda()
   
   if self.is_train:
   	self.optimizer = self.train_config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.train_config.learning_rate)
   ```

##### 5.2 def train()

1. 定义loss函数，调用到 `functions.py` (7)中的 `DiffLoss, MSE, SIMSE, CMD`：

   ```Python
   if self.train_config.data == "ur_funny":
       self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
   else: # mosi and mosei are regression datasets
       self.criterion = criterion = nn.MSELoss(reduction="mean")
       
   self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
   self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
   self.loss_diff = DiffLoss()
   self.loss_recon = MSE()
   self.loss_cmd = CMD()
   ```

2. 开始 epoch 训练：
   
   * 通过 `self.model.train()` 启用 BatchNormalization 和 Dropout。
   
   * 开始每个 batch 训练：
   
     * `self.model.zero_grad()`：把模型中参数的梯度设为 0。
   
     * 张量都进行 `to_gpu()` 操作。
   
     * `self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)` 调用模型的 `forward()` (6.2)函数：
   
       ```Python
       y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
       ```
   
     * 分别求各种 loss：
   
       ```Python
       cls_loss = criterion(y_tilde, y)  # MSE loss
       diff_loss = self.get_diff_loss()
       domain_loss = self.get_domain_loss()
       recon_loss = self.get_recon_loss()
       cmd_loss = self.get_cmd_loss()
       ```
   
     * 每个 loss 乘以各自的权重系数后相加，得到最终的总 loss：
       `diff_weight`：0.3；`sim_weight`：1.0；`recon_weight`：1.0
   
       ```python
       if self.train_config.use_cmd_sim:
           similarity_loss = cmd_loss
       else:
           similarity_loss = domain_loss
       
       loss = cls_loss + self.train_config.diff_weight * diff_loss + \
           self.train_config.sim_weight * similarity_loss + \
           self.train_config.recon_weight * recon_loss
       ```
   
     * 反向传播：
   
       ```Python
       loss.backward()
                       
       torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
       self.optimizer.step()
       ```
   
   * 每个 epoch 进行一次 **验证**：
   
     ```Python
     valid_loss, valid_acc = self.eval(mode="dev")
     ```
   
   * 根据验证集上的 loss 来保存效果最好的模型。6轮 epoch 后如果 loos 不再降低，就通过 `curr_patience <= -1` 这个条件来加载之前保存的最好的一次模型，然后根据 `num_trials <= 0` 这个条件退出训练：
   
     ```Python
     print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
     if valid_loss <= best_valid_loss:
         best_valid_loss = valid_loss
         print("Found new best model on dev set!")
         if not os.path.exists('checkpoints'):
             os.makedirs('checkpoints')
     	torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
         torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
         curr_patience = patience
     else:
         curr_patience -= 1
         if curr_patience <= -1:
             print("Running out of patience, loading previous best model.")
             num_trials -= 1
             curr_patience = patience
             self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
             self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
             lr_scheduler.step()
             print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
     
     if num_trials <= 0:
         print("Running out of patience, early stopping.")
         break
     ```
   
3. 进行 **test**：

   ```Python
   self.eval(mode="test", to_print=True)
   ```

##### 5.3 def eval(mode=None, to_print=False)

1. `model.eval()`：不启用 BatchNormalization 和 Dropout。

2. 加载数据：

   ```Python
   if mode == "dev":
       dataloader = self.dev_data_loader
   elif mode == "test":
       dataloader = self.test_data_loader
   
       if to_print:
           self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
   ```

3. 每个batch 进行 dev 或 test：

   ```Python
   with torch.no_grad():  # 一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
       for batch in dataloader:
           self.model.zero_grad()
           t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
   
           t = to_gpu(t)
           v = to_gpu(v)
           a = to_gpu(a)
           y = to_gpu(y)
           l = to_gpu(l)
           bert_sent = to_gpu(bert_sent)
           bert_sent_type = to_gpu(bert_sent_type)
           bert_sent_mask = to_gpu(bert_sent_mask)
   
           y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
   
           if self.train_config.data == "ur_funny":
               y = y.squeeze()
   
   		cls_loss = self.criterion(y_tilde, y)
           loss = cls_loss  # 只用求分类loss
   
           eval_loss.append(loss.item())
           y_pred.append(y_tilde.detach().cpu().numpy())
           y_true.append(y.detach().cpu().numpy())
   ```

##### 5.4 def get_diff_loss()

* 特定空间和共享空间之间的差异性：

  ```Python
  def get_diff_loss(self):
      shared_t = self.model.utt_shared_t
      shared_v = self.model.utt_shared_v
      shared_a = self.model.utt_shared_a
      private_t = self.model.utt_private_t
      private_v = self.model.utt_private_v
      private_a = self.model.utt_private_a
  
      # Between private and shared
      loss = self.loss_diff(private_t, shared_t)
      loss += self.loss_diff(private_v, shared_v)
      loss += self.loss_diff(private_a, shared_a)
  
      # Across privates
      loss += self.loss_diff(private_a, private_t)
      loss += self.loss_diff(private_a, private_v)
      loss += self.loss_diff(private_t, private_v)
  
      return loss
  ```

##### 5.5 def get_recon_loss()

* 模型中 `Decoder` 部分最后会求重构损失：

  ```Python
  def get_recon_loss(self, ):
      loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)  # MSE loss
      loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
      loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
      loss = loss/3.0
      return loss
  ```

##### 5.6 def get_cmd_loss()

* `n_moments` 参数设置为 `5`，三个模态两两之间求一次 `cmd_loss`，然后求算术平均：

  ```Python
  def get_cmd_loss(self,):
      if not self.train_config.use_cmd_sim:
          return 0.0
  
      # losses between shared states
      loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
      loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
      loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
      loss = loss/3.0
  
      return loss
  ```

  

#### 6 models.py

* 由于不考虑使用 `BERT`，所以这里不看涉及到 `BERT` 的文本编码部分。

##### 6.1 初始化-定义

1. 文本、图像和语音的编码器均使用了 `RNN` 。

   ```Python
   self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
   self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
   self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
   
   self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
   self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
   
   self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
   self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)
   ```

2. 将三种模态映射到相同尺寸 `config.hidden_size` 的空间：

   ```Python
   self.project_t = nn.Sequential()
   self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
   self.project_t.add_module('project_t_activation', self.activation)
   self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
   
   self.project_v = nn.Sequential()
   self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
   self.project_v.add_module('project_v_activation', self.activation)
   self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
   
   self.project_a = nn.Sequential()
   self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
   self.project_a.add_module('project_a_activation', self.activation)
   self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))
   ```

3. private 编码器：

   ```Python
   self.private_t = nn.Sequential()
   self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
   
   self.private_v = nn.Sequential()
   self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
   
   self.private_a = nn.Sequential()
   self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
   ```

4. 共享编码器：

   ```Python
   self.shared = nn.Sequential()
   self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.shared.add_module('shared_activation_1', nn.Sigmoid())
   ```

5. 重构：

   ```Python
   self.recon_t = nn.Sequential()
   self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.recon_v = nn.Sequential()
   self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.recon_a = nn.Sequential()
   self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   ```

6. shared space adversarial discriminator，共享空间对抗辨别器：

   ```Python
   self.discriminator = nn.Sequential()
   self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
   self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
   self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
   self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))
   ```

7. shared-private collaborative discriminator：

   ```Python
   self.sp_discriminator = nn.Sequential()
   self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))
   ```

8. 融合层：

   ```Python
   self.fusion = nn.Sequential()
   self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
   self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
   self.fusion.add_module('fusion_layer_1_activation', self.activation)
   self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))
   ```

9. LayerNorm：

   ```Python
   self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
   self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
   self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
   ```

10. Transformer编码器：

    ```python
    encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    ```

##### 6.2 def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)

* **有个问题，`lengths` 这些参数是怎么传过来的？？？**
  data_loader中传过来的？

1. 调用 `alignment()` 函数。

   ```Python
   def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
       batch_size = lengths.size(0)
       o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
       return o
   ```

##### 6.3 def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)

1. 分别从文本、图像和语音模态提取特征：均调用了 `extract_features()` 函数，用双层 RNN 来提取各自模态的特征。

   ```Python
   # extract features from text modality
   sentences = self.embed(sentences)
   final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
   utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
   
   # extract features from visual modality
   final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
   utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
   
   # extract features from acoustic modality
   final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
   utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
   ```
   
   * `permute(dims)`：将 tensor 的维度换位。
   * `contiguous()`：为什么需要这个方法？因为 `torch.view` 等方法操作需要连续的Tensor。transpose、permute 操作虽然没有修改底层一维数组，但是新建了一份 Tensor 元信息，并在新的元信息中的 重新指定 stride。torch.view 方法约定了不修改数组本身，只是使用新的形状查看数据。如果我们在 transpose、permute 操作后执行 view，Pytorch 会抛出错误。

2. 在 `shared_private()` 函数中调用之前定义好的 `project、private 和 shared` 等模块。

   ```python
   def shared_private(self, utterance_t, utterance_v, utterance_a):
       # Projecting to same sized space
       self.utt_t_orig = utterance_t = self.project_t(utterance_t)
       self.utt_v_orig = utterance_v = self.project_v(utterance_v)
       self.utt_a_orig = utterance_a = self.project_a(utterance_a)
   
       # Private-shared components
       self.utt_private_t = self.private_t(utterance_t)
       self.utt_private_v = self.private_v(utterance_v)
       self.utt_private_a = self.private_a(utterance_a)
   
       self.utt_shared_t = self.shared(utterance_t)
       self.utt_shared_v = self.shared(utterance_v)
       self.utt_shared_a = self.shared(utterance_a)
   ```

3. 调用之前定义的 shared-private collaborative discriminator：

   ```Python
   self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
   self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
   self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
   self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
   ```

4. 在 `reconstruct()` 函数中调用之前定义好的 `recon` 模块：

   ```Python
   def reconstruct(self,):
       self.utt_t = (self.utt_private_t + self.utt_shared_t)
       self.utt_v = (self.utt_private_v + self.utt_shared_v)
       self.utt_a = (self.utt_private_a + self.utt_shared_a)
   
       self.utt_t_recon = self.recon_t(self.utt_t)
       self.utt_v_recon = self.recon_v(self.utt_v)
       self.utt_a_recon = self.recon_a(self.utt_a)
   ```

5. Transformer 融合层：

   ```Python
   # 1-LAYER TRANSFORMER FUSION
   h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
   h = self.transformer_encoder(h)
   h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
   o = self.fusion(h)
   ```

   * `self.transformer_encoder()` 是初始化(6.1-10)时定义好的编码层。
   * `self.fusion()` 是初始化(6.1-8)时定义好的融合层。

##### 6.4 def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm)

1. 调用 `torch.nn.utils.rnn` 中的 `pack_padded_sequence`：

   参数：

   > `input`：经过 `pad_sequence()` 处理之后的数据，在本模型中，`pad_sequence()` 是在 `data_loader.py` 的 `get_loader()` 中使用；
   >
   > `lengths`：mini-batch中各个序列的实际长度；
   >
   > `batch_first`：True 对应 [batch_size, seq_len, feature] ，False 对应 [seq_len, batch_size, feature] ；
   >
   > `enforce_sorted`：如果是 True ，则输入应该是按长度降序排序的序列。如果是 False ，会在函数内部进行排序。默认值为 True 。

   Q：为什么要使用这个函数？
   A：因为在进行 forward 计算时，会把 padding_value 也考虑进去，可能会导致 RNN 通过了非常多无用的 padding_value，这样不仅浪费计算资源，最后得到的值可能还会存在误差。为了使 RNN 可以高效的读取数据进行训练，就需要在 pad 之后再使用 pack_padded_sequence 对数据进行处理。默认条件下，必须把输入数据按照序列**长度从大到小**排列后才能送入 pack_padded_sequence ，否则会报错。

2. 具体代码：

   ```Python
   def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
       packed_sequence = pack_padded_sequence(sequence, lengths)
   
       if self.config.rnncell == "lstm":
           packed_h1, (final_h1, _) = rnn1(packed_sequence)
   	else:
   		packed_h1, final_h1 = rnn1(packed_sequence)
   
   	padded_h1, _ = pad_packed_sequence(packed_h1)
       normed_h1 = layer_norm(padded_h1)
       packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
   
       if self.config.rnncell == "lstm":
           _, (final_h2, _) = rnn2(packed_normed_h1)
   	else:
           _, final_h2 = rnn2(packed_normed_h1)
   
   	return final_h1, final_h2
   ```

   

#### 7 functons.py

* 文件内定义了各个损失函数的计算方法。

##### 7.1 class MSE(nn.Module)

1. 均方误差：

   ```Python
   class MSE(nn.Module):
       def __init__(self):
           super(MSE, self).__init__()
   
       def forward(self, pred, real):
           diffs = torch.add(real, -pred)
           n = torch.numel(diffs.data)
           mse = torch.sum(diffs.pow(2)) / n
   
           return mse
   ```

##### 7.2 class DiffLoss(nn.Module)

1. 改变两个输入的维度：

   ```Python
   batch_size = input1.size(0)
   input1 = input1.view(batch_size, -1)
   input2 = input2.view(batch_size, -1)
   ```

2. 分别与各自的平均值相减：

   ```Python
   # Zero mean
   input1_mean = torch.mean(input1, dim=0, keepdims=True)
   input2_mean = torch.mean(input2, dim=0, keepdims=True)
   input1 = input1 - input1_mean
   input2 = input2 - input2_mean
   ```

3. 求L2范数：

   ```Python
   input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
   input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
   
   input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
   input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
   ```

   * `torch.norm(input, p, dim, out=None,keepdim=False) → Tensor`：返回输入张量给定维 `dim` 上每行的 `p` 范数。 
     参数：
   
     > input (Tensor) – 输入张量
     >
     > p (float) – 范数计算中的幂指数值
     >
     > dim (int) – 缩减的维度
     >
     > out (Tensor, optional) – 结果张量
     >
     > keepdim（bool）– 保持输出的维度 （此参数官方文档中未给出，但是很常用）
   
   * `expand_as()`：将输入tensor的维度扩展为与指定tensor相同的size。

4. 求均值平方：

   ```Python
   diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
   ```

   * `mean()` 函数的参数：dim=0，按行求平均值，返回的形状是（1，列数）；dim=1,按列求平均值，返回的形状是（行数，1），默认不设置dim的时候，返回的是所有元素的平均值。

##### 7.3 class CMD(nn.Module)【重点！！！】

1. Central Moment Discrepancy (CMD)，中心矩差异。公式：
   $$
   CMD_K(X,Y)=\frac{1}{|b-a|}||E(X)-E(Y)||_2+\sum_{k=2}^K{\frac{1}{|b-a|^k}||C_k(X)-C_k(Y)||_2} \\
   where \quad E(X)=\frac{1}{|X|}\sum_{x\in X}{x},\quad C_k(X)=E((x-E(X))^k)
$$
   `E(x)` 是样本 X 的经验期望向量；`Ck(X)` 是
   
2. 参数：x1，x2，n_mom ents。在本实验中 `n_moments` 设置为 `5`。【这里设置成5有什么意义吗？】

3. `forward()` 函数：

   ```Python
   def forward(self, x1, x2, n_moments):
       # 求两个输入的平均值
       mx1 = torch.mean(x1, 0)
       mx2 = torch.mean(x2, 0)
       # 原始输入减去自身平均值
       sx1 = x1-mx1
       sx2 = x2-mx2
       # 求L2范数 ||E(X)-E(Y)||2
       dm = self.matchnorm(mx1, mx2)
       scms = dm
       for i in range(n_moments - 1):
           scms += self.scm(sx1, sx2, i + 2)
   	return scms
   ```

4.  `matchnorm()`函数：求L2范数 `||E(X)-E(Y)||_2`。

   ```Python
   def matchnorm(self, x1, x2):
       power = torch.pow(x1-x2,2)
       summed = torch.sum(power)
       sqrt = summed**(0.5)
       return sqrt
   ```

5. `scm()` 函数：

   ```Python
   def scm(self, sx1, sx2, k):
       ss1 = torch.mean(torch.pow(sx1, k), 0)
       ss2 = torch.mean(torch.pow(sx2, k), 0)
       return self.matchnorm(ss1, ss2)
   ```

   