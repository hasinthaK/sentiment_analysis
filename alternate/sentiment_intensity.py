import string
import xlrd
from xlutils.copy import copy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

xl_file = xlrd.open_workbook('ml - Copy.xlsx')
xl_file_scored = copy(xl_file)
sheet_scored = xl_file_scored.get_sheet(0)
sheet = xl_file.sheet_by_index(0)

# sheet_scored.write(0, 0, 'Review')
sheet_scored.write(0, 1, 'Sentiment')
sheet_scored.write(0, 2, 'Score')


def get_sentiment(value):
    if value > 0:
        return 'Positive'
    elif value < 0:
        return 'Negative'
    else:
        return 'Neutral'


def write_excel(score, index):
    print(index, score)
    compound = score['compound']
    sheet_scored.write(index, 1, get_sentiment(compound))
    sheet_scored.write(index, 2, format(compound, ".2%"))


def analyze(text, index):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    write_excel(score, index)


for i in range(sheet.nrows-1):
    text = sheet.cell_value(i+1, 0)

    lower_case = text.lower()
    cleaned_text = lower_case.translate(
        str.maketrans("", "", string.punctuation))

    analyze(cleaned_text, i+1)

xl_file_scored.save("ml_scored.xls")
