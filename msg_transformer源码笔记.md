### MSG-Transformer
[TOC]
#### 1 config.py
1. `yacs`库：yacs是一个轻量级的用来管理系统配置参数的库。
* 头文件：
  
    > `from yacs.config import CfgNode as CN   `
* 创建容器：
    > `_C = CN()`
    > `_C.model = CN()  # 嵌套使用`
* clone()：克隆一份配置节点_C的信息返回，_C的信息不会改变。
  
    > `config = _C.clone()`
* merge_from_file()：对于不同的实验，你有不同的超参设置，所以你可以使用yaml文件来管理不同的configs，然后使用`merge_from_file()`这个方法，这个会比较每个experiments特有的config和默认参数的区别，会将默认参数与特定参数不同的部分，用特定参数覆盖。
  
    > `config.merge_from_file("./test_config.yaml")`
* freeze()：冻结配置，该操作后不能修改值。
  
    > `config.freeze()`
* defrost()：`freeze()`的反操作。
  
    > `config.defrost()`

#### 2 logger.py
1. `logging.getLogger([name=None])`：指定name，返回一个名称为name的Logger实例。
2. `Logger.setLevel()`：设置日志级别。
3. `Logger.addHandler()` 和 `Logger.removeHandler()`：添加和删除一个Handler。
4. format常用格式说明：
    > %(levelno)s: 打印日志级别的数值
    > %(levelname)s: 打印日志级别名称
    > %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
    > %(filename)s: 打印当前执行程序名
    > %(funcName)s: 打印日志的当前函数
    > %(lineno)d: 打印日志的当前行号
    > %(asctime)s: 打印日志的时间
    > %(thread)d: 打印线程ID
    > %(threadName)s: 打印线程名称
    > %(process)d: 打印进程ID
    > %(message)s: 打印日志信息
5. 日志输出-控制台：
    * `logging.StreamHandler()`：创建一个handler，用于输出到控制台。
    * `Handler.setFormatter(formatter)`
6. 日志输出-文件：
   
    * `logging.FileHandler(logfile, mode='w')`：mode参数有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志；a是追加模式，默认如果不写的话，就是追加模式。

#### 3 data\build.py
##### 3.1 build_loader
##### 3.2 build_dataset
##### 3.3 build_transform
1. `timm.data.create_transform()`：**【只要是图片都要经过这个操作吗？】**
2. `torchvision.transforms.RandomCrop(size, padding)`：依据给定的 size 随机裁剪。例如padding=4，则上下左右均填充4个pixel


#### 4 main.py
1. 分布式初始化：
`torch.distributed.init_process_group(backend, init_method, world_size, rank)`
    * `backend` 参数可以参考 [PyTorch Distributed Backends](https://pytorch.org/docs/master/distributed.html?highlight=distributed#backends)，也就是分布式训练的底层实现，GPU 用 `nccl`，CPU 用 `gloo`。  
    如果利用nccl在每个机器上使用多进程，每个进程必须独占访问它使用的每个GPU，因为在进程间共享GPU将会导致停滞
    * `init_method` 参数就是多进程通信的方式。
    * `world_size` 参数就是全局进程个数。
2. `torch.distributed.barrier()`：不同进程之间的数据同步。
3. 线性scaled学习率设置：**【为什么是除以512？】**
```python
linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * 
dist.get_world_size() / 512.0linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * 
config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * 
dist.get_world_size() / 512.0
# gradient accumulation also need to scale the learning rate
if config.TRAIN.ACCUMULATION_STEPS > 1:   
    linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS    
    linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS    
    linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
```
4. 调用`logger.py`中的`create_logger()`。
5. 调用`data\build.py`中的`build_lodader()`。
6. 调用`model\bulid.py`中的`build_model.py(config)`，`build_model(config)`调用`msg_transformer.py`中的`MSGTransformer(...)`。
7. 调用 `optimizer.py` 中的 `build_optimizer(...)`。

#### 5 msg_transformer.py
##### 5.1 class MSGTransformer(nn.Module)
1. 参数：
> `img_size (int | tuple(int))`: Input image size. Default 224
> `patch_size (int | tuple(int))`: Patch size. Default: 4
> `in_chans (int)`: Number of input image channels. Default: 3
> `num_classes (int)`: Number of classes for classification head. Default: 1000
> `embed_dim (int)`: Patch embedding dimension. Default: 96
> `depths (tuple(int))`: Depth of each MSG-Transformer layer.
> `num_heads (tuple(int))`: Number of attention heads in different layers.
> `window_size (int)`: Window size. Default: 7
> `mlp_ratio (float)`: Ratio of mlp hidden dim to embedding dim. Default: 4，
> mlp hidden dim/embedding dim的值
> `qkv_bias (bool)`: If True, add a learnable bias to query, key, value. Default: True
> `qk_scale (float)`: Override default qk scale of head_dim ** -0.5 if set. Default: None
> `drop_rate (float)`: Dropout rate. Default: 0
> `attn_drop_rate (float)`: Attention dropout rate. Default: 0
> `drop_path_rate (float)`: Stochastic depth rate. Default: 0.1
> `norm_layer (nn.Module)`: Normalization layer. Default: nn.LayerNorm.
> `ape (bool)`: If True, add absolute position embedding to the patch embedding. Default: False
> `patch_norm (bool)`: If True, add normalization after patch embedding. Default: True
> `shuffle_size (list(int))`: shuffle region size of each stage
> `manip_type (str)`: the operation type for manipulating msg tokens: shuf or none
2. self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

3. split image into non-overlapping patches：
   
    * 调用 `PatchEmb(nn.Module)`（5.2）。
    
4. absolute position embedding：
    * `nn.Parameter()`：可以把这个函数理解为类型转换函数，将一个不可训练的类型 `Tensor`转换成可以训练的类型 `parameter` 并将这个 `parameter` 绑定到这个 `module` 里面。
    * `timm.models.layers.trunc_normal_(tensor, std)`：目的是用**截断的正态分布**绘制的值填充输入张量。
    
5. 定义 `msg/-tokens`：

    ```Python
    self.msg_tokens = nn.Parameter(torch.zeros(1, shuffle_size[0]**2, 1, embed_dim))
    trunc_normal_(self.msg_tokens, std=.02)
    ```

6. 随机深度【不是太理解这里】：

    * `torch.linspace(start, end, steps, out=None) → Tensor`：返回一个1维张量，包含在区间start和end上均匀间隔的step个点。输出张量的长度由steps决定。

7. `manip_op = shuffel_msg()`，函数具体代码：

    ```Python
    def shuffel_msg(x):
        # (B, G, win**2+1, C)
        B, G, N, C = x.shape
        if G == 1:
            return x
        msges = x[:, :, 0] # (B, G, C)
        assert C % G == 0
        msges = msges.view(-1, G, G, C//G).transpose(1, 2).reshape(B, G, 1, C)
        x = torch.cat((msges, x[:, :, 1:]), dim=2)
        return x
    ```

    维度变换：

    `(B, G, win**2+1, C)` -> `(B, G, C)` -> `(B, G, G, C//G)` -> `(B, G, C//G, G)` -> `B, G, 1, C`

    然后再和 `x[:, :, 1:]` concat还原诚 `(B, G, win**2+1, C)`。

8. 调用`BasicLayer(nn.Module)`（5.3）。

9. 调用 `nn.LayerNorm` 和 `nn.Linear`。

10. 通过 `self.apply()` 调用初始化权重函数 `_init_weights()`：

    ```Python
    self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    	elif isinstance(m, nn.LayerNorm):
    		nn.init.constant_(m.bias, 0)
    		nn.init.constant_(m.weight, 1.0)
    ```



##### 5.2 class PatchEmbed(nn.Module)

1. 将图像转换为 Patch Embedding。
##### 5.3 class BasicLayer(nn.Module)

1. 调用`MSGBlock(nn.Module)`（5.4）。
2. 下采样 `downsample` 就是`PatchMerging`（5.6）。

##### 5.4 class MSGBlock(nn.Module)

1. 调用`WindowAttention(nn.Module)`。
2. 通过一层`norm_layer`；再通过一层`MLP`。

##### 5.5 class WindowAttention(nn.Module)

1. 【分割窗口那里没看懂】

##### 5.6 class PatchMerging(nn.Module)

1. 把 `msg_token` 分割成4份，然后在最后一维上进行 `cat`。【其实有点没懂这里转换shape的作用】
2. `msg_token` 再通过 `LayerNorm` 和线性层。
3. `x` 也是同样步骤。
4. 然后 `msg_token` 和 `x` 再连接起来。



