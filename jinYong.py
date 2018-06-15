import tensorflow as tf
import os

import dataPreProcess

def load_jinyongData():
    text = ''
    for dirname in os.listdir('./data/jinyong'):
        for filename in os.listdir(os.path.join('./data/jinyong',dirname)):
            if filename.endswith('txt'):
                text_temp = []
                with open(os.path.join('./data/jinyong/' + dirname, filename), 'rb') as f:
                    text_data = f.read()
                text_data = text_data.decode("gbk")
                text_temp = text_data.split('\r\n')
                text_temp[:4] = ''
                text = text + ''.join(text_temp)


    return text

text = load_jinyongData()
print(len(text))
# print(text)
lines_of_text = text.split('\u3000\u3000')
print(len(lines_of_text))
print(lines_of_text[:100])
# lines_of_text = lines_of_text[14:]
# print(lines_of_text)
lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]
print(len(lines_of_text))
print(lines_of_text[:20])
lines_of_text = [lines.strip() for lines in lines_of_text]
print(len(lines_of_text))
print(lines_of_text[:20])
lines_of_text = dataPreProcess.two_otherNoUseContent(lines_of_text)
print(len(lines_of_text))
print(lines_of_text[:20])