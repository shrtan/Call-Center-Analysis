from pandas import read_csv, datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from autocomplete import predict

#Custom format for the dates
def parser(x):
return datetime.strptime('190'+str(x), '%Y-%m')

def read_file(filename):
series = read_csv(filename, header=0, parse_dates=['Date'], index_col=0, squeeze=True)
print series.head()


#Plot the data
series.plot()
pyplot.show()

#Autocorrelation plot
autocorrelation_plot(series)
pyplot.show()

#Partial autocorrelation function
plot_pacf(series)
pyplot.show()


#Fitting the arima model
model = ARIMA(series, order=(5,1,0)) #Lag of 5, first order differencing
model_fit = model.fit(disp=0) #disp=0 to display no debug errors
print model_fit.summary()

#Plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print residuals.describe()


X = series.values #All the sales values
#size = int(len(X) * 0.66) #Define train and test data size
train, test = X[0:424], X[424:484] #Divide dataset into train and test
history = [x for x in train] #History initially contains all the train data
predictions = list() #List of all the predictions made

#model = ARIMA(history, order=(5,1,0))
#model_fit = model.fit(disp=0)

#For all the test dates (future) predict
for t in range(len(test)):
print "1"
model = ARIMA(history, order=(30,1,0)) #Recompute the model on the new set of dates (active learning)
model_fit = model.fit(disp=0)
output = model_fit.forecast() #Predict the next values
print "2"
predict = output[0] #Take the next value
predictions.append(predict) #Add that to the prediction list
obs = test[t] #True sales value
history.append(predict) #Append the true value to the history
print 'predicted = ', predict, "expected = ", obs



error = mean_squared_error(test, predictions) 
print 'Test MSE: ', error

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


def forecast(filename):
read_file(filename)

if __name__ == '__main__':
filename = 'Daily_Logs.csv'
forecast(filename)
