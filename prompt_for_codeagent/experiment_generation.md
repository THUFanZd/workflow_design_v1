# 任务描述
我想要实现一个SAE特征解释的workflow，能够给出一个SAE特征的激活(即 输入端)和干预解释(即 输出端)。我现在想完成这个workflow的第3步。

# workflow

1. 从neuronpedia上获取初始观察
2. 根据初始观察，生成解释假说
3. 针对假说，设计测例

# 细节描述
## 前置环节
前置环节的代码已经完成。你需要获得initial_hypothesis_generation.py中的generate_initial_hypotheses函数的返回结果，获得要输入给LLM API的解释文本。
## 设计实验
这一环节，你需要调用LLM API，基于输入的假说，分别给出SAE特征激活和干预假设的验证实验
1. 工作方式、输入、输出
对于输入端和输出端，分别给每个假说设计测例。
对于输入端，调用LLM API，将已有的解释假说（generate_initial_hypotheses函数的返回结果是一个dict，其中的input_side_hypotheses字段是需要提供给这个环节的LLM API的输入端假说，output_side_hypotheses字段是需要提供给这个环节LLM API的输出端假说）提供给LLM API，对于每个解释，根据解释的字面，生成n个句子，使得这n个句子能够激活这个解释对应的SAE特征。这个n是需要作为函数参数。返回一个list，每个元素对应generate_initial_hypotheses的input_side_hypotheses中的一个假说，为针对该假说设计的假说。
对于输出端，对于每个假说，无需设计，都先只返回列表["The explanation is simple:", "I think", "We"]。
2. 设计原则
两端生成的函数是同一个，通过一个参数控制是生成 输入端还是输出端 解释假说的测例
prompt放到prompts/experiments_generation_prompt.py
prompt用英文写
prompt的设计：不能太简陋，要讲清任务背景，输入端的测例要激活SAE特征。
# 实现要求
涉及LLM API的使用时，不要实现heuristic的方法
核心代码放到一个单独的文件当中
在if name == main中，添加一些代码，使得我可以直接通过运行这个文件，测试这个模块
需要保存解释原文，设计的测例，LLM API的输入和输出，到base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts这个目录下，创建一个md文件保存。这个path在generate_initial_hypotheses中有，因为生成初始解释假说需要调用这个函数，所以自然地将目录和它保持一致
需要统计对话的token用量。统计的功能在function.py中有。如果需要，可以参考initial_hypothesis_generation.py中统计token用量的方法。