import pickle
import os

# 加载数据
def load_text(path):
    input_file = os.path.join(path)

    with open(input_file, 'r') as f:
        text_data = f.read()

    return text_data

# 第一层处理：去掉空格和换行符
def first_spaceAndEnter(text):
    num_words_for_training = 100000

    text = text[:num_words_for_training]
    # print(text)
    lines_of_text = text.split('\n')
    # print(len(lines_of_text))
    # print(lines_of_text[:15])
    lines_of_text = lines_of_text[14:]
    # print(lines_of_text)
    lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]
    # print(len(lines_of_text))
    # print(lines_of_text[:20])
    lines_of_text = [lines.strip() for lines in lines_of_text]
    # print(lines_of_text)
    return lines_of_text

# 去掉无用的内容和嵌在书中的广告
def two_otherNoUseContent(lines_of_text):
    import re
    # 生成一个正则，负责找『[]』包含的内容
    pattern = re.compile(r'\[.*\]')
    # 将所有指定内容替换成空
    lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
    # print(lines_of_text)
    # 将上面的正则换成负责找『<>』包含的内容
    pattern = re.compile(r'<.*>')
    # 将所有指定内容替换成空
    lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
    # 将上面的正则换成负责找『......』包含的内容
    pattern = re.compile(r'\.+')
    # 将所有指定内容替换成空
    lines_of_text = [pattern.sub("。", lines) for lines in lines_of_text]
    # 将上面的正则换成负责找行中的空格
    pattern = re.compile(r' +')
    # 将所有指定内容替换成空
    lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]
    # print(lines_of_text[:20])
    # print(lines_of_text[-20:])
    # 将上面的正则换成负责找句尾『\\r』的内容
    pattern = re.compile(r'\\r')
    # 将所有指定内容替换成空
    lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
    # print(lines_of_text[-20:])
    return lines_of_text

# 文字 《==》 数字
def create_lookup_tables(input_data):
    vocab = set(input_data)
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab

# 符号 《==》 字母
def token_lookup():
    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])

    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]

    return dict(zip(symbols, tokens))


# 将文字转换格式内容存储preprocess.p
def preprocess_and_save_data(text, token_lookup, create_lookup_tables):
    token_dict = token_lookup()

    for key, token in token_dict.items():
        text = text.replace(key, '{}'.format(token))

    text = list(text)

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]

    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('./data/preprocess.p', 'wb'))


def load_preprocess():
    return pickle.load(open('./data/preprocess.p', mode='rb'))


def save_params(params):
    pickle.dump(params, open('./data/params.p', 'wb'))


def load_params():
    return pickle.load(open('./data/params.p', mode='rb'))
