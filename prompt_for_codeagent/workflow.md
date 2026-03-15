# 任务描述
我想要实现一个SAE特征解释的workflow，能够给出一个SAE特征的激活(即 输入端)和干预解释(即 输出端)。我现在想完成这个workflow的第7步。
# workflow

1. 从neuronpedia上获取初始观察
2. 根据初始观察，生成解释假说
3. 针对假说，设计测例
4. 进行实验，得到实验结果
5. 将实验结果存储到记忆当中
6. 根据记忆，以及当前轮次的实验结果，对假说进行修正、更改等
7. 重复步骤3-6，直到解释假说收敛，或者达到最大轮次
# 细节描述
## 前置环节
前置环节的代码已经完成。
## 功能要求
这一环节，你需要将整个workflow，用一个py文件串联起来，作为整个workflow的运行器
基础功能：你需要在main中，调用前序步骤对应的函数，将模块串联起来，完成对SAE特征输入、输出侧的解释。
设置最大迭代轮数，默认1轮（迭代轮数的定义是，refine的调用次数）
最终给出 输入端和输出端的解释假说
附加功能：
首先，你需要对之前的模块进行一个修改。之前模块保存结果，都是保存在base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts目录下，现在，我需要在ts下再加一层目录，为round_id，先进行这个修改
然后，由于每个环节运行都需要消耗API的tokens，需要实现功能：指定从哪一轮的哪一步开始真正运行模块函数，之前的步骤，仿照前序步骤模块的main中reuse-from-logs的实现，从文件中直接读取需要的内容，之后再调用模块函数。寻找之前的实验结果的逻辑跟之前一样，在ts目录下找，如果没找到就报错。
## 实现要求
代码单独放到一个py文件当中
输入侧和输出侧的假说优化函数是同一个，通过参数控制针对的是输入侧还是输出侧。这两侧的大致工作原理是相同的，最大的区别还是LLM API的prompt的内容
在if name == main中，添加一些代码，使得我可以直接通过运行这个文件，测试这个模块。
注意前序模块的函数不要在该部分的函数的代码中调用，而是在main中，直接调用前序模块的函数，从其返回结果中抽取需要的信息作为参数，输入到该模块的函数中。（将这种效果称为解耦）
前序模块的函数在main中的调用，可以参考experiments_generation.py
运行该模块代码的main，需要保存memory的内容，到base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts这个目录下，创建一个md文件保存。这个path在generate_initial_hypotheses中都有（前序模块应该都有）
参考之前的环节，在main中实现--reuse-from-logs
实验执行模块的实现在方便合理的前提下，也可以解耦
涉及LLM API的使用时，不要实现heuristic的方法
需要保存workflow最终的输入端和输出端的假说，到base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts / final_result 这个目录下，创建一个md文件保存。这个path在generate_initial_hypotheses等前序模块中有，跟前序模块保持一致
需要统计workflow对话的token用量。统计的功能在function.py中有。如果需要，可以参考initial_hypothesis_generation.py中统计token用量的方法。
function.py中的函数如果可以复用则复用
# 已有文件讲解
functions.py中实现了调用llm api，解析json文件等函数，你需要先查看其中的内容，看有没有可以直接复用的函数。
model_with_sae.py中实现了待解释LLM和SAE的前向传播代码
neuronpedia_feature_api.py对应初始观察获取
initial_hypothesis_generation.py对应初始假说生成环节
experiments_generation.py对应实验设计
experiments_execution.py对应实验执行，返回对应的结果
hypothesis_memory.py对应当前轮次的记忆总结
hypothesis_refinement.py对应假说优化