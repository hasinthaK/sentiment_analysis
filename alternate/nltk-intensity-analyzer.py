import string
from collections import Counter
# excel file utils
import xlrd
from xlutils.copy import copy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

xl_file = xlrd.open_workbook('ml.xlsx')
# xl_file_copy = copy(xl_file)
sheet = xl_file.sheet_by_index(0)


def analyze(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    print(score)


for i in range(sheet.nrows):
    text = sheet.cell_value(i, 0)
    # text = open("read.txt", encoding="utf-8").read()

    lower_case = text.lower()
    cleaned_text = lower_case.translate(
        str.maketrans("", "", string.punctuation))

    analyze(cleaned_text)

    # tokenized_words = word_tokenize(cleaned_text, "english")

    # final_words = []
    # for word in tokenized_words:
    #     if word not in stopwords.words('english'):
    #         final_words.append(word)

    # emotions_list = []
    # with open('emotions.txt', 'r') as file:
    #     for line in file:
    #         cleared_line = line.replace('\n', '').replace(
    #             ',', '').replace("'", '').strip()
    #         word, emotion = cleared_line.split(':')
    #         if word in final_words:
    #             emotions_list.append(emotion)
    # print(emotions_list)
    # w = Counter(emotions_list)
    # print(w)


# xl_file_copy.save("ml_copy.xlsx")
