import numpy as np
from io import open
import unicodedata
import string
import re
import random

import time
import math


def single_text_To_two_text(sour_text_path, ques_text_path, ans_text_path):

    with open(sour_text_path, 'r', encoding='utf-8') as f:

        with open(ques_text_path, encoding='utf-8', mode='w') as qus_f:

            with open(ans_text_path, encoding='utf-8', mode='w') as ans_f:


                for i in f.readlines():

                    line = i.split('\t')
                    ques = line[0]
                    qus_f.write(ques + '\n')


                    ans = line[1]
                    ans_f.write(ans)



    pass












if __name__ == '__main__':

    path = 'chatterbot.tsv'
    single_text_To_two_text(path, 'question', 'answer')


