# 任务描述
我想要实现一个SAE特征解释的workflow，能够给出一个SAE特征的激活(即 输入端)和干预解释(即 输出端)。现在，我需要你实现其中的第2步

前两步：
1.从neuronpedia上获取初始观察
2.根据初始观察，生成解释假说

# 细节描述
## 从neuronpedia上获取初始观察
这一环节的代码已经完成，你需要调用neuronpedia_feature_api.py中的fetch_and_parse_feature_observation函数，获得要输入给LLM API的初始观察对应的dict。
## 根据初始观察生成假说
注：调用LLM API的方法见./llm_api下的方法和信息，你可以调用我的LLM API，但是要注意控制用量尽量小；涉及LLM API的使用时，不要实现heuristic的方法
这一个环节，你需要调用LLM API，基于输入的观察，分别给出SAE特征的初始的激活和干预假设
1.工作方式
提供一个参数（num_hypothesis，简称n），控制生成假说的数量。
对于输入端和输出端，分别调用LLM API，各自给出假说。
设置两种给出假说的方式。通过一个参数调整生成方式
第一种：直接让一个LLM 一次性生成n个假说。即调用一次LLM API
第二种：让一个LLM 生成一个假说，然后让后面的LLM，除了能看到输入的初始观察，还能看到前面的LLM 生成的假说，然后生成更健全/多样/不同角度的假说。也就是说，要调用n次LLM API
2.输入：
a.对于激活端，从fetch_and_parse_feature_observation的返回值中，取input_side_obseravtion键的值，作为要放到prompt里的观察；对于干预端，取output_side_obseravtion键的值，作为要放到prompt里的观察。
b.针对激活和干预，分别设计prompt，引导LLM基于观察，输出对SAE特征的解释。两个prompt的结构要大致一样。
3.输出：
两端各有一个List，保存了两端各n个假说
4.设计原则
两端生成的函数是同一个，通过一个参数控制是生成 输入端还是输出端 假说
prompt可以放到特定目录下的文件中。目前的实现只是一个workflow中的一个环节，后续的prompt需要也能放到这个目录下。（比如prompts/hypothesis_generation_prompt.py）
prompt用英文写
prompt的设计：不能太简陋，要讲清任务背景，输入输出观察是怎么来的，LLM 应该基于观察输出解释，解释要控制在30词内能够清晰地引导LLM进行输出
5.其他要求
核心代码放到一个单独的文件当中
在if name == main中，添加一些代码，使得我可以直接通过运行这个文件，测试这个模块
需要保存生成假说的数量，生成假说的方式，LLM API的输入和输出，到base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts这个目录下，创建一个md文件保存。这个path在fetch_and_parse_feature_observation中有，因为生成初始解释需要调用这个函数，所以自然地将目录和它保持一致
需要统计对话的token用量(可以参考api_use_example.py，流式下是可以统计token数的，不过那里统计了单词对话的输入输出和总token，你需要统计的是初始假说生成全过程的输入输出和总token数。像统计token数这种比较通用的功能，可以写到function.py中，使得其他环节需要的时候也可以直接调用)


你需要补充对输入和输出侧观察内容的介绍。
输入侧：
包含了激活句子数目和对应的句子。每个activation example包含了句子文本，非0激活值和对应的token，以及最大激活值和对应的token。
输出侧：
将SAE解码器中对应的特征向量，用LLM的unembedding layer映射到对应的token上，得到token logits的最大最小者。
另外，不用说这些观察来自neuronpedia