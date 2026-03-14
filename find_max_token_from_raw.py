import json
# 打开feature_raw.json文件
with open('feature_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

first_activation_sentence_dict = data["activations"][0]
max_token = first_activation_sentence_dict["tokens"][first_activation_sentence_dict["maxValueTokenIndex"]]
print(max_token)
