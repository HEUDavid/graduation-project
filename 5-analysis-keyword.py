import os
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.font_manager as fm
from pylab import *
import pandas as pd
import jieba, re, numpy
from pyecharts import WordCloud



# 读数据
def read_csv():
    excel_data = pd.read_excel("DG1124-1124.xlsx").to_dict(orient = "records")
    content = ""
    for i in excel_data:
        content = content + i["微博内容"]
    return content

# 文本处理
def cleanStr(content):
    # 去除所有评论里多余的字符
    content = content.replace(" ", ",")
    content = content.replace(" ", "、")
    content = re.sub('[,，。. \r\n]', '', content)
    return content

# 分词
def cutWords(content):
    comment = cleanStr(content)
    comment = jieba.cut(content,cut_all=False)
    wl_space_split = " ".join(comment)
    return wl_space_split

import jieba.analyse
from PIL import Image,ImageSequence
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator




data = read_csv()

result=jieba.analyse.textrank(data,topK=50,withWeight=True)
keywords = dict()
for i in result:
    keywords[i[0]]=i[1]
print(keywords)


image= Image.open('dg.jpg')
graph = np.array(image)
wc = WordCloud(font_path='./fonts/simhei.ttf',background_color='White',max_words=400,mask=graph,width=1000,height=1000)
wc.generate_from_frequencies(keywords)
image_color = ImageColorGenerator(graph)
plt.imshow(wc)
plt.imshow(wc.recolor(color_func=image_color))
plt.axis("off")
plt.show()
wc.to_file('dg-cloud.jpg')







