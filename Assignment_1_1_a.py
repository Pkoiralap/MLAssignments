import matplotlib.pyplot as plt

X = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
Y = [5.1,6.1,6.9,7.8,9.2,9.9,11.5,12.0,12.8]

#Y = a + bX
#           ∑y∑x^2 - ∑x ∑xy
#  a = __________________________
#           n∑x^2 - (∑x)^2

#           n∑xy - ∑x∑xy
#  b = __________________________
#           n∑x^2 - (∑x)^2


# Simple linear regression

n = len(X)
sumY = sum(Y)
sumX = sum(X)
sumX2 = sum([i**2 for i in X])
sumXY = sum([x*y for x,y in zip(X,Y)])

denominator = float(n * sumX2 - sumX * sumX)
a = (sumY*sumX2 - sumX*sumXY) / denominator
b = (n * sumXY - sumX * sumY) / denominator
predicted_y_values = [a + b * x for x in X]


plt.scatter(X, Y, label="Input Data")
plt.plot(X, predicted_y_values, color='r', label='Prediction Line')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.legend()

plt.show()

