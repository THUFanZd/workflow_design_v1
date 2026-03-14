你需要参考的是neuronpedia_feature_api.py中的fetch_feature_json函数。

该函数从neuronpedia网站上返回一个json文件，包含了该特征的所有信息。其内容可以参考feature_raw.json。

你需要完成以下任务：
1. 将该json文件存储到logs/{layer_id}_{feature_id}/{timestamp}/目录下，文件名为layer{layer_id}-feature{feature_id}-neuronpedia-raw.json
2. 从中抽取以下信息：
  - "neg_str"和"neg_values"
  - "pos_str"和"pos_values"
  - "activations"
3. 信息解析：
  1. 输入侧信息解析：
    "activations"是一个数组，每个元素是一个dict，代表了一个激活特征的句子以及相伴随的一些信息，数组中的元素按照最大激活值降序排序。其中，最大激活值，是这句话中所有token的最大激活值。
    你需要设置一个参数，决定抽取元素的方法。
    方法1：
    一共抽取m+n个元素
    阶段1：抽取这个数组的前m个元素
    阶段2：然后再抽取n个元素。
    后面这n个元素的抽取方式为：从第m+1个元素开始，查找dict中激活值最大的token（参考find_max_token_from_raw.py），要求和上一个dict中的最大的token不同（严格匹配，区分大小写）。终止条件为，抽取满n个元素，或者按照这种方法，"activations"中找不够n个元素。如果是后者，则缺多少元素，就从第m+1个元素开始抽多少个，但是要避开已经抽取到的元素。（比如m=5,n=5，抽取了1,2,3,4,5,8号元素，发现抽取不到10个元素，就再把6,7,9,10放进来）。如果还是抽不到n个元素就返回。
    该方法接受两个参数，分别是阶段1抽取的元素数m，和阶段2抽取的元素数n。
    方法2：
    按照方法1的抽取后n个元素的方式，从第1个元素开始，抽取够n个元素即可。如果不够，也直接返回。该方法复用方法1的第二个参数n
    方法3：
    直接抽取前m个元素。该方法复用方法1的第一个参数m
  2. 输出侧信息解析
    将pos_str和pos_value一一配对，neg_str和neg_value一一配对即可。
4. 输出格式
用一个dict返回结果，其中包含：
layer_id和feature_id
输入侧观察
输出侧观察
每个部分对应一个key，再在这些key下组织你解析到的信息

5. 其他要求：
将这个dict存储到logs/{layer_id}_{feature_id}/{timestamp}/目录下，文件名为layer{layer_id}-feature{feature_id}-observation-input.json
实现功能的函数添加到neuronpedia_feature_api.py中。并在该py文件的if name == main部分运行以上函数。


我已经把中文key换成英文了，并做了一点小改动，你就以现在的代码为基础就行。
进行以下变更：
1. 在fetch_and_parse_feature_observation的基础上，添加一个函数，叫convert_to_input_observation，对其返回值，做以下信息筛选：
  保留fetch parse函数返回的dict中，input_side_observation和output_side_observation键
  对于input_side_observation，只保留selected_count和activations
  对于activations中的每个元素，做如下操作：
    构造"sentence"字段，作用是将tokenslist中的tokens拼接成原来的那句话
    构造"activation_tokens"字段，是一个List[dict]，每一个元素包含了这句话中，激活值不为0的token以及对应的激活值
    保留maxValue和max_token字段
  activations键的名字换为activation_examples

  依旧保存fetch parse函数的输出，其输出文件名我已经更改。而你现在实现的函数的输出也要保存，路径同fetch parse函数，文件路径为observation_path = base_dir / f"layer{layer_id}-feature{feature_id}-observation-input.json"

2. 程序运行参数有一些调整和疑问：
  1. index和feature_id有区别吗？如果没有，把index删掉
  2. source参数开头的数字是layer_id，结尾是width，默认16k，所以这个参数的头和尾用另外的这两个参数解析。中间默认gemmascope
  3. model id 默认gemma-2-2b, selection-method默认1，m和n默认5
  4. api-key改成neuronpedia-api-key
  