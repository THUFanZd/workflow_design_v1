# 任务描述
我想要实现一个SAE特征解释的workflow，能够给出一个SAE特征的激活(即 输入端)和干预解释(即 输出端)。我现在想完成这个workflow的第4步。
# workflow

1. 从neuronpedia上获取初始观察
2. 根据初始观察，生成解释假说
3. 针对假说，设计测例
4. 进行实验，得到实验结果
# 细节描述
## 前置环节
前置环节的代码已经完成。你需要获得experiments_generation.py中的generate_hypothesis_experiment函数返回的实验用例。
## 运行实验
这一环节，你需要用输入的假说，分别做实验，给出SAE特征激活和干预假设的验证指标计算结果
1. 输入、工作方式、输出
输入：取自generate_hypothesis_experiment函数返回的dict
输入端/激活实验：输入的是dict的"input_side_experiments"
输出端/干预实验：输入的是dict的"output_side_experiments"
工作方式：
输入实验：对于一个假说的每一个测例，将他们输入到model_with_sae中，进行前向传播，传播到SAE，提取对应特征的激活值，计算这些测例的非0率，作为这个输入端假说的得分。
输出实验：参考feature_descriptions_pipeline.ipynb中的代码的实现，对于四个KL散度，寻找对应干预强度，然后将测例输入LLM+SAE，并用干预强度干预SAE对应特征的激活值，将输出的结果保存下来。然后调用LLM API，做m（可以用参数控制，默认是3）选1判断，判断次数可以通过参数设置，默认为1. 将判断正确率作为输出端假说的得分。
注：explanation_quality_evaluation/output-side-evaluation目录下的intervention_blind_score.py是我在别处实现的一个测评方式，似乎跟我在这里对工作方式的描述相同，你需要先确认一下，然后可以使用其中可用的代码。不过生成的代码还是放到项目目录下，也不要从intervention_blind_score.py中调用函数，如果有需要则直接拿到你的代码中。
另外，该目录下有一些txt文件，在真正有运行结果前，你可以用这些txt文件中的内容作为对照组的输入内容，也可以参考这些文件的格式，作为你实验函数输出的格式
2. prompt设计原则
prompt放到prompts/experiments_execution_prompt.py
prompt用英文写
prompt的设计：不能太简陋，要讲清任务背景，输入端的测例要激活SAE特征。
## 实现要求
涉及LLM API的使用时，不要实现heuristic的方法
两端的实验代码可以分别放到两个py文件当中
在if name == main中，添加一些代码，使得我可以直接通过运行这个文件，测试这个模块。
注意前序模块的函数不要在该部分的函数的代码中调用，而是在main中，直接调用前序模块的函数，从其返回结果中抽取需要的信息作为参数，输入到该模块的函数中。（将这种效果称为解耦）
前序模块的函数在main中的调用，可以参考experiments_generation.py
实验执行模块的实现在方便合理的前提下，也可以解耦
需要保存解释原文，测例，实验结果，得分，LLM API的输入和输出，到base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts这个目录下，创建一个md文件保存。这个path在generate_initial_hypotheses中有，因为生成初始解释假说需要调用这个函数，所以自然地将目录和它保持一致
需要统计对话的token用量。统计的功能在function.py中有。如果需要，可以参考experiments_generation.py中统计token用量的方法。
# 已有文件讲解
functions.py中实现了调用llm api，解析json文件等函数，你需要先查看其中的内容，看有没有可以直接复用的函数。
model_with_sae.py中实现了待解释LLM和SAE的前向传播代码
neuronpedia_feature_api.py对应初始观察获取
initial_hypothesis_generation.py对应初始假说生成环节
python_command.txt保存了可以运行该项目的环境的路径