from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier as rfc
import pandas as pd
# import xlrd

# xl_file = xlrd.open_workbook('ml.xlsx')
# sheet = xl_file.sheet_by_index(0)

# text = ""

# for i in range(sheet.nrows):
#     text = sheet.cell_value(i, 0) + text


def openFile(path):
    with open(path) as file:
        data = file.read()
    return data


imdb_data = openFile('data_sets/imdb_labelled.txt')
amzn_data = openFile('data_sets/amazon_cells_labelled.txt')
yelp_data = openFile('data_sets/yelp_labelled.txt')

datasets = [imdb_data, amzn_data, yelp_data]

combined_dataset = []
# separate samples from each other
for dataset in datasets:
    combined_dataset.extend(dataset.split('\n'))

# separate each label from each sample
dataset = [sample.split('\t') for sample in combined_dataset]

df = pd.DataFrame(data=dataset, columns=['Reviews', 'Labels'])

# Remove any blank reviews
df = df[df["Labels"].notnull()]


vectorizer = TfidfVectorizer(min_df=15)
bow = vectorizer.fit_transform(df['Reviews'])
labels = df['Labels']

selected_features = SelectKBest(chi2, k=200).fit(
    bow, labels).get_support(indices=True)

vectorizer = TfidfVectorizer(min_df=15, vocabulary=selected_features)

bow = vectorizer.fit_transform(df['Reviews'])


X_train, X_test, y_train, y_test = train_test_split(
    bow, labels, test_size=0.33)

classifier = rfc()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
