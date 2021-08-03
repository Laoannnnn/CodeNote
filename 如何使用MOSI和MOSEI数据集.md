### 如何使用MOSI和MOSEI数据集

* 参考[MISA源码](https://github.com/declare-lab/MISA_)

#### 1 MOSI/MOSEI

* **方法1**：可以选择通过 `load_pickle` 函数直接导入 `.pkl` 文件。
* **方法2**：

1. 构建 `word2id` 索引，第一行在python里是可行的，这样构建出来的字典，当加入新的单词的时候，`value` 会自增：

   ```Python
   word2id = defaultdict(lambda: len(word2id))
   UNK = word2id['<unk>']
   PAD = word2id['<pad>']
   print(word2id)  # {'<pad>': 1, '<unk>': 0}
   
   # turn off the word2id - define a named function here to allow for pickling
   def return_unk():
       return UNK
   ```

2. 定义一些加载函数：

   ```Python
   def to_pickle(obj, path):
       with open(path, 'wb') as f:
           pickle.dump(obj, f)
   
   def load_pickle(path):
       with open(path, 'rb') as f:
           return pickle.load(f)
       
   from tqdm import tqdm_notebook
   def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
       if init_emb is None:
           emb_mat = np.random.randn(len(w2i), embedding_size)
       else:
           emb_mat = init_emb
       f = open(path_to_embedding, 'r')
       found = 0
       for line in tqdm_notebook(f, total=embedding_vocab):
           content = line.strip().split()
           vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
           word = ' '.join(content[:-300])
           if word in w2i:
               idx = w2i[word]
               emb_mat[idx, :] = vector
               found += 1
       print(f"Found {found} words in the embedding file.")
       return torch.tensor(emb_mat).float()
   ```

3. 构建保存数据集路径并选取数据集：

   ```Python
   from mmsdk import mmdatasdk as md
   
   if not os.path.exists(DATA_PATH):
       check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)
   DATASET = md.cmu_mosi
   ```

4. 载入高级特征、原始数据和标签：

   ```Python
   md.mmdataset(DATASET.highlevel, DATA_PATH)  # high-level features
   md.mmdataset(DATASET.raw, DATA_PATH)  # low-level(raw) data
   md.mmdataset(DATASET.labels, DATA_PATH)  # label
   ```

5. 定义不同的模态：

   ```Python
   # define your different modalities - refer to the filenames of the CSD files
   visual_field = 'CMU_MOSI_VisualFacet_4.1'
   acoustic_field = 'CMU_MOSI_COVAREP'
   text_field = 'CMU_MOSI_TimestampedWords'
   label_field = 'CMU_MOSI_Opinion_Labels'
   
   features = [text_field, visual_field, acoustic_field]
   ```

6. 构建数据集：

   ```Pyton
   recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
   dataset = md.mmdataset(recipe)
   ```

7. 文本对齐：

   ```Python
   # we define a simple averaging function that does not depend on intervals
   def avg(intervals: np.array, features: np.array) -> np.array:
       try:
           return np.average(features, axis=0)
       except:
           return features
   
   # first we align to words with averaging, collapse_function receives a list of functions
   dataset.align(text_field, collapse_functions=[avg])
   ```

8. 标签对齐：

   ```Python
   # we add and align to lables to obtain labeled segments
   # this time we don't apply collapse functions so that the temporal sequences are preserved
   label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
   dataset.add_computational_sequences(label_recipe, destination=None)
   dataset.align(label_field)
   ```

8. 分割 `train/dev/test` 集：

   ```Python
   # obtain the train/dev/test splits - these splits are based on video IDs
   train_split = DATASET.standard_folds.standard_train_fold
   dev_split = DATASET.standard_folds.standard_valid_fold
   test_split = DATASET.standard_folds.standard_test_fold
   ```

9. 开始进行分段：

   ```Python
   # define a regular expression to extract the video ID out of the keys
   pattern = re.compile('(.*)\[.*\]')
   num_drop = 0  # a counter to count how many data points went into some processing issues
   
   for segment in dataset[label_field].keys():
       # 1. 首先分割出三个模态
       # get the video ID and the features out of the aligned dataset
       # serch把结果分为几组, group(0)返回匹配正则表达式整体结果, group(1)返回下标第0组
       vid = re.search(pattern, segment).group(1)
       label = dataset[label_field][segment]['features']
       _words = dataset[text_field][segment]['features']
       _visual = dataset[visual_field][segment]['features']
       _acoustic = dataset[acoustic_field][segment]['features']
       
       # if the sequences are not same length after alignment, there must be some problem with some modalities
       # we should drop it or inspect the data again
       if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
           print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
           num_drop += 1
           continue
           
   	# remove nan values
       label = np.nan_to_num(label)
       _visual = np.nan_to_num(_visual)
       _acoustic = np.nan_to_num(_acoustic)
       
       # 2. 处理单词/图像/语音
       # remove speech pause tokens - this is in general helpful
       # we should remove speech pauses and corresponding visual/acoustic features together
       # otherwise modalities would no longer be aligned
       actual_words = []
       words = []
       visual = []
       acoustic = []
       for i, word in enumerate(_words):
           if word[0] != b'sp':
               actual_words.append(word[0].decode('utf-8'))
               # SDK stores strings as bytes, decode into strings here
               words.append(word2id[word[0].decode('utf-8')]) 
               visual.append(_visual[i, :])
               acoustic.append(_acoustic[i, :])
   
   	words = np.asarray(words)
   	visual = np.asarray(visual)
   	acoustic = np.asarray(acoustic)
       
       # Z-score归一化
       # z-normalization per instance and remove nan/infs
       visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
       acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
       
       if vid in train_split:
           train.append(((words, visual, acoustic, actual_words), label, segment))
   	elif vid in dev_split:
           dev.append(((words, visual, acoustic, actual_words), label, segment))
       elif vid in test_split:
           test.append(((words, visual, acoustic, actual_words), label, segment))
       else:
   		print(f"Found video that doesn't belong to any splits: {vid}")
           
   	# 设置default_factory, 这里没太看懂为什么这么做？
       word2id.default_factory = return_unk
   ```

10. 返回的数据集的形式：每个数据记录内容为 `(words, visual, acoustic, actual_words), label, segment`。



