import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import numpy

def parser(x):
    return pd.to_datetime.strptime('19'+x, '%Y-%m')
series = pd.read_csv('C:\\Users\\mahim\\OneDrive\\Desktop\\Minor-2\\trend_dataset - Sheet1.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
# fit linear model
X = [i for i in range(0, len(series))]
X = numpy.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
pyplot.plot(y)
pyplot.plot(trend)
pyplot.show()
# detrend
detrended = [y[i]-trend[i] for i in range(0, len(series))]
# plot detrended
pyplot.plot(detrended)
pyplot.show()
