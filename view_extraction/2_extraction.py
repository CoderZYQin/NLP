import re
import jieba
import os

def get_pog(sentence):

    LTP_DATA_DIR = '/Users/wangnan/Desktop/Artificial/2_IBM/1_Ltp'

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

    from pyltp import Postagger

    postagger = Postagger()

    postagger.load(pos_model_path)

    postags = postagger.postag(sentence)

    postagger.release()

    return postags

def get_ner(sentence, postags):

    LTP_DATA_DIR = '/Users/wangnan/Desktop/Artificial/2_IBM/1_Ltp'

    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')

    from pyltp import NamedEntityRecognizer

    recongnizer = NamedEntityRecognizer()

    recongnizer.load(ner_model_path)

    netags = recongnizer.recognize(sentence, postags)

    recongnizer.release()

    return netags

def get_parser(sentence, postags):

    LTP_DATA_DIR_S = '/Users/wangnan/Desktop/Artificial/2_IBM/1_Ltp'

    parser_model_path = os.path.join(LTP_DATA_DIR_S, 'parser.model')

    from pyltp import Parser

    parser = Parser()

    parser.load(parser_model_path)

    arcs = parser.parse(sentence, postags)

    parser.release()

    return arcs

# 1. 先找到目标句子
def split_sentences(text):
    text = text.replace('\n', '')
    sentences = re.split('(。|！|\!|\.|？|\?)', text)
    new_sents = []
    for i in range(int(len(sentences)/2)):
        sent = sentences[2*i] + sentences[2*i+1]
        new_sents.append(sent)
    return new_sents

def get_target_sentence(sentences, keywords):
    target_sentences = []
    for sent in sentences:
        words = jieba.cut(sent)
        for word in words:
            if word in keywords:
                target_sentences.append(sent)
    return target_sentences

# 2. 找到观点表述人
def get_elaborater(sentence):

    cut_words = list(jieba.cut(sentence))

    postags = get_pog(cut_words)

    arcs = get_parser(cut_words, postags)

    for index, value in enumerate(arcs):
        if value.relation == 'SBV':
            return cut_words[index]

# 3. 找到人物言论
def get_views(sentence, kewords):
    cut_words = list(jieba.cut(sentence))

    postags = get_pog(cut_words)

    arcs = get_parser(cut_words, postags)

    hed_index = 0

    # 如果HED没有在关键词里面, 就舍弃该句子
    for index, value in enumerate(arcs):
        if value.relation == 'HED':
            hed_index = index
            if cut_words[index] not in kewords:
                return 0

    netags = get_ner(cut_words, postags)

    name = ''
    name_list = []

    # 根据NER寻找主语, 如果找不到, 就舍弃该句子
    for index, value in enumerate(netags):
        if 'Nh' in value:
            if arcs[index].relation == 'SBV':
                if value[0] == 'S':
                    name = cut_words[index]
                elif value[0] == 'B':
                    name_list.append(index)
                elif value[0] == 'E':
                    name_list.append(index)

    if name_list:
        offset = name_list[0]
        while offset <= name_list[-1]:
            name += cut_words[offset]
            offset += 1
    else:
        if not name:
            return 0

    if arcs[hed_index + 1].relation == 'WP':
        hed_index += 2

    viewpoints = ''.join(cut_words[hed_index:])

    return (name, viewpoints)



# def get_speaker_and_points(sentences, keywords):
#     # target_sentences = get_target_sentence(split_sentences(sentences), keywords)
#     # for sentence in target_sentences:


if __name__ == '__main__':

    with open('news_content.txt', 'r', encoding='utf-8') as f:
        news_content = f.read()

    say_words = ['表示', '指出', '认为', '坦言', '看来', '透露', '介绍', '明说', '说', '强调', '所说', '提到', '说道', '称', '声称', '建议', '呼吁',
                 '提及', '地说', '直言', '普遍认为', '批评', '重申', '提出', '明确指出', '觉得', '宣称', '猜测', '特别强调', '写道', '引用', '相信',
                 '解释', '谈到', '深知', '称赞', '感慨', '主张', '还称', '中称', '指责', '披露', '明确提出', '描述', '提醒', '深有体会', '爆料',
                 '裁定', '宣布']


    # print(get_elaborater('法官马提亚·齐根 星期四早些时候对法庭说，在这种情况下，这项裁决不会立即生效。'))

    print(get_views('法官马提亚·齐根星期四早些时候对法庭说，在这种情况下，这项裁决不会立即生效。', say_words))

    # get_speaker_and_points(news_content, say_words)