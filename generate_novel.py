import tensorflow as tf
import numpy as np
import dataPreProcess
# import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = dataPreProcess.load_preprocess()
seq_length, load_dir = dataPreProcess.load_params()


def get_tensors(loaded_graph):
    inputs = loaded_graph.get_tensor_by_name("inputs:0")

    initial_state = loaded_graph.get_tensor_by_name("init_state:0")

    final_state = loaded_graph.get_tensor_by_name("final_state:0")

    probs = loaded_graph.get_tensor_by_name("probs:0")

    return inputs, initial_state, final_state, probs


def pick_word(probabilities, int_to_vocab):
    # chances = []
    #
    # for idx, probs in enumerate(probabilities):
    #     for prob in probs:
    #         if prob >= 0.05:
    #             chances.append(int_to_vocab[idx])
    #
    # rand = np.random.randint(0, len(chances))
    # return str(chances[rand])
    num_word = np.argmax(probabilities)
    # num_word = np.random.choice(len(int_to_vocab), p=probabilities)
    # return num_word
    return int_to_vocab[num_word]

# 生成文本的长度
gen_length = 500

# 文章开头的字，指定一个即可，这个字必须是在训练词汇列表中的
prime_word = '我'


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载保存过的session
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # 通过名称获取缓存的tensor
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # 准备开始生成文本
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # 开始生成文本
    for n in range(gen_length):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[0][dyn_seq_length - 1], int_to_vocab)
        gen_sentences.append(pred_word)

    # 将标点符号还原
    novel = ''.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '（', '“'] else ''
        novel = novel.replace(token.lower(), key)
    # novel = novel.replace('\n ', '\n')
    # novel = novel.replace('（ ', '（')

    print(novel)