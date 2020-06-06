from analyzer_module import sentiment
# excel file utils
import xlrd
from xlutils.copy import copy

# change the file name accordingly
xl_file = xlrd.open_workbook('ml - Copy.xlsx')
sheet = xl_file.sheet_by_index(0)
xl_file_scored = copy(xl_file)
sheet_scored = xl_file_scored.get_sheet(0)

sheet_scored.write(0, 1, 'Sentiment')
sheet_scored.write(0, 2, 'Score')


def write_excel(sentiment_value, score, index):
    sheet_scored.write(index, 1, sentiment_value)
    sheet_scored.write(index, 2, score)


def analyze(text, index):
    v, score = sentiment(text)
    write_excel(v, score, index)


# read excel file
for i in range(sheet.nrows-1):
    text = sheet.cell_value(i+1, 0)
    analyze(text, i+1)

# save created excel file
xl_file_scored.save("ml_scored.xls")
