import matplotlib.pyplot as plt

X = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
Y = [5.1,6.1,6.9,7.8,9.2,9.9,11.5,12.0,12.8]

def grad_theta_1(X, Y, theta_1, theta_2):
    return sum([theta_1 + theta_2 * x - y for x,y in zip(X, Y)])

def grad_theta_2(X, Y, theta_1, theta_2):
    return sum([theta_1 * x + theta_2 * x*x - x*y for x,y in  zip(X, Y)])

def trainModel(X,Y, theta_1, theta_2, alpha, acceptance_criteria, max_tries):
    J_array, i = [], 0
    while True:
        # calculate the J value and append in an array for plotting later on
        J = 1.0/(2*len(X)) * sum([ 
            (y - ( theta_1 + theta_2 * x )) ** 2  
                for x,y in zip(X,Y)
        ])
        J_array.append(J)

        # calculate the change in theta values
        theta_1_change = alpha * grad_theta_1(X, Y, theta_1, theta_2)
        theta_2_change = alpha * grad_theta_2(X, Y , theta_1, theta_2)
        if (
            (abs(theta_1_change) <= acceptance_criteria and abs(theta_2_change) < acceptance_criteria) or
            i > max_tries
        ):
            break

        # apply the change in theta values
        theta_1 = theta_1 - theta_1_change
        theta_2 = theta_2 - theta_2_change
        i += 1
    return (J_array, theta_1, theta_2)


# model parameters
num_of_iterations = 10**10
theta_1 = 0
theta_2 = 1
acceptance_criteria = 10**-10

(predictedJs_1, theta_1_1, theta_2_1) = trainModel(X, Y, theta_1, theta_2, 0.01, acceptance_criteria, num_of_iterations)
(predictedJs_2, _, _) = trainModel(X, Y, theta_1, theta_2, 0.005, acceptance_criteria, num_of_iterations)
(predictedJs_3, _, _) = trainModel(X, Y, theta_1, theta_2, 0.001, acceptance_criteria, num_of_iterations)
predicted_y_values = [theta_1_1 * theta_2_1*x for x in X]


# plots
_, (ax1,ax2) = plt.subplots(2,1, sharex=False, sharey=False)
ax1.plot(predictedJs_1, label="J value alpha = 0.0001")
ax1.plot(predictedJs_2, label="J value alpha = 0.005")
ax1.plot(predictedJs_3, label="J value alpha = 0.001")
ax1.set(xlabel="Iterations", ylabel="J Value")


ax2.scatter(X, Y, label="Input Data")
ax2.plot(X, predicted_y_values, color='r', label='Prediction Line')
ax2.set(xlabel="X", ylabel="Y")

ax2.legend()
ax1.legend()

title = f"J values and prediction lines for {num_of_iterations} iterations"
plt.title(title, pad=200)
plt.show()
