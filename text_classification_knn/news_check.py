import pandas as pd

content = pd.read_csv('../sqlResult_1558435.csv', encoding='gb18030')

print(content.head(5))