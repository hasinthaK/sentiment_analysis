# Sentiment analysis

Script analyzes given sentences & provide the calculated sentiment value,
& then store in an excel file.  
The initial data to be analyzed (sentences) are to be read from an excel file.

**All input file names must be exact same as denoted here**

File name to be read:
```ml - Copy.xlsx```    
Note: This file should be in exact name & format internally to get the correct result.

then the results are written to file:
```ml_scored.xls```

**Example files included**

## Working

1. run ```settings.py``` from ```/support``` directory.
2. download all the nltk packages by selecting 'all' from the window.
3. train the models using any of the desired trainers named as ```trainer#.py``` in the root directory.
-- Note: All the trained modles will be saved as byte files in 'trained' directory.
4. run ```main.py``` .

**Output will be written to a newly created file in the root directory as ```ml_scored.xls```**

#
**Current test results**

*Best results were gained after using ```trainer2.py``` for initial training.   
(Use trainer ```sentiment_intensity.py``` from 'alternate' directory if neither of ```trainer#.py``` outputs desired results.)