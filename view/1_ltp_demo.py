# -*- coding: utf-8 -*-

import os

# 词性标注
LTP_DATA_DIR = '/Users/wangnan/Desktop/Artificial/2_IBM/1_Ltp'

pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

from pyltp import Postagger

postagger = Postagger()

postagger.load(pos_model_path)

words = ['元芳', '你', '怎么', '看']

postags = postagger.postag(words)

print('\t'.join(postags))

postagger.release()

# 命名实体识别

LTP_DATA_DIR = '/Users/wangnan/Desktop/Artificial/2_IBM/1_Ltp'

ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')

from pyltp import NamedEntityRecognizer

recongnizer = NamedEntityRecognizer()
recongnizer.load(ner_model_path)

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
netags = recongnizer.recognize(words, postags)

print('\t'.join(netags))

# 释放模型
recongnizer.release()

# 依存句法分析

LTP_DATA_DIR = '/Users/wangnan/Desktop/Artificial/2_IBM/1_Ltp'

parser_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')

from pyltp import Parser

parser = Parser()

parser.load(parser_model_path)

words = ['元芳', '你', '怎么', '看']

postags = ['nh', 'r', 'r', 'v']

arcs = parser.parse(words, postags)

print('\t'.join('%d:%s'% (arc.head, arc.relation) for arc in arcs))

# 释放模型
parser.release()