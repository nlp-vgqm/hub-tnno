#coding:utf-8

import os
import json
from zai import ZhipuAiClient

'''
利用Agent完成快速学习：
（1）学习内容简介，构造学习线路（如何学习，要掌握那些内容，初步目标，升级目标，终极目标），并推荐相关书籍、paper、公众号等资料，推荐快速入门、详细讲解等资料。
（2）注入学习资料，生成目录、简介这样的结构化文档（多级标题，不超过3级）。如果原文已有结构，就对标题进行提炼。如果原文没有结构，就进行文本匹配，归在某一个标题下或者新生成一个标题。可以转成字典数据、json文件等。
'''

#pip install zhipuai
#https://open.bigmodel.cn/ 注册获取APIKey
def call_large_model(prompt):
    client = ZhipuAiClient(api_key=os.environ.get("zhipuApiKey")) #key填入环境变量
    response = client.chat.completions.create(
        model="glm-3-turbo",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text

def summarize(content):
    # 向大模型提问想要学习的一个知识点的简介
    summary_prompt = f"请简单介绍{content}主要是学习什么，包括哪些重要内容："
    summary = call_large_model(summary_prompt)
    print(summary)
    return theme_analysis_result

def road(content):
    #向大模型提问，学习一个知识点的学习方法/学习路线
    road_prompt=f'想学习{content}相关内容，请制定一套学习计划。对于每一个学习阶段，请附上学习主要内容、完成目标和大概用时。'
    road = call_large_model(road_prompt)
    print(road)
    return road

def recommend(content):
    #向大模型提问，推荐相关学习资料（包括书籍、视频、公众号等）
    recommendation_prompt=f'想学习{content}相关内容，请推荐一些学习资料，包括书籍、论文、主页、视频和公众号等。如果是书籍、论文请尽量提供下载链接。如果是主页、视频，请提供相关链接。如果是公众号，请提供公众号的名字。'
    recommendation = call_large_model(recommendation_prompt)
    print(recommendation)
    return recommendation

def structured(text_path):
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()
    #对于任何一篇文章，不管是长文章，还是短文章，生成结构化结构
    structure_text_json={}
    structure_text_prompt=f'请依据输入文本{text}，输出结构化目录。其中标题不超过三级，比如1级标题序号为1.，二级标题序号为1.1，三级标题序号为1.1.1。标题序号后空格再输出标题内容。标题内容请进行概括，字数不长于15个字。每一个标题（不管是哪一级标题）之后请换行。'
    #得重新写一个上传文件的方式
    structure_text=call_large_model(structure_text_prompt)    #输出一大段字符串。一个标题换一行
    title_list = structure_text.split('\n')
    order=''
    content=''
    for title in title_list:
        order , content=title.split(' ')
        structure_text_json.get(order,content)

    with open('index.json', 'w', encoding='utf-8') as f:
        json.dump(structure_text_json, f, ensure_ascii=False, indent=4)
    return structure_text_json

def faststudy_agent(content):
    #自定义工作流，先不做问题匹配和function call匹配，这里直接调用function call
    summary_content=summarize(content)
    method = road(content)
    recommendation = recommend(content)
    print(summary_content)
    print(method)
    print(recommendation)
    flag = input('是否需要针对上传文档进行结构化处置：Y/N')
    if flag == 'Y':
        path = input('请输入文档的完整路径和文档名（包括扩展名）：')
        structure_text = structured(path)
        print(structure_text)
    else:
        pass


if __name__ == "__main__":
    faststudy_agent('Agent开发')


