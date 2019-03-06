file_base = '../3_Spider_data'

import os

print(len(os.listdir(file_base)))

import jieba

def cut(string): return ' '.join(jieba.cut(string))

corpus = [
  cut(open(os.path.join(file_base, f))) for f in os.listdir(file_base)
]

