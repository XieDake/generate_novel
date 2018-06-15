import numpy as np
import tensorflow as tf
import dataPreProcess

#############################################################
# the batches
#############################################################
def get_batches(int_text, batch_size, seq_length):
    # 计算有多少个batch可以创建
    # n_batches = (len(int_text) // (batch_size * seq_length))

    # 计算每一步的原始数据，和位移一位之后的数据
    # batch_origin = np.array(int_text[: n_batches * batch_size * seq_length])
    # batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    # 将位移之后的数据的最后一位，设置成原始数据的第一位，相当于在做循环
    # batch_shifted[-1] = batch_origin[0]

    # batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)
    # batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)

    # batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))

    characters_per_batch = batch_size * seq_length
    num_batches = len(int_text) // characters_per_batch

    # clip arrays to ensure we have complete batches for inputs, targets same but moved one unit over
    input_data = np.array(int_text[: num_batches * characters_per_batch])
    # target? 方式不一样？
    target_data = np.array(int_text[1: num_batches * characters_per_batch + 1])

    inputs = input_data.reshape(batch_size, -1)
    targets = target_data.reshape(batch_size, -1)

    inputs = np.split(inputs, num_batches, 1)
    targets = np.split(targets, num_batches, 1)

    batches = np.array(list(zip(inputs, targets)))
    batches[-1][-1][-1][-1] = batches[0][0][0][0]

    return batches

int_text, vocab_to_int, int_to_vocab, token_dict = dataPreProcess.load_preprocess()
# print(int_text)
batches = get_batches(int_text, 128, 32)