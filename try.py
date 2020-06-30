# 尝试通过json文件来显示visdom，目前是失败的。
import visdom
import json

# 保存现有env
str1 = 'try'
str2 = 'luck'
vis = visdom.Visdom(env=str1+' '+str2)    # 默认使用的env是main
vis.text('Hello, world')            # 打印些字符
vis.save([str1+' '+str2])

# 读取env
with open("C:/Users/73416/.visdom/try luck.json", "r") as f:
    predata = json.load(f)
