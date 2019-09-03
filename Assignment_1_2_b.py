import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def grad(X, Y, theta_vec):
    return [
        - 1.0/len(X) * sum( 
            [ (y - x.dot(theta_vec)) * x[j] for (x,y) in zip(X,Y) ]
        ) for j in range(len(theta_vec))
    ]

def cost(X, Y, theta_vec):
    return 1.0/(2*len(X)) * sum([
            (y - x.dot(theta_vec)) ** 2 for x,y in zip(X,Y)
        ])

def train_model(X, Y, theta_vec, learning_rate, convergence_criteria, max_iterations):
    i, j_array = 0, []
    while True:
        j_array.append(cost(X,Y, theta_vec))

        grad_vec = grad(X, Y ,theta_vec)
        new_theta = [theta - learning_rate * grad_vec[i] for i,theta in enumerate(theta_vec)]

        if (
            i > max_iterations or
            abs(cost(X,Y, theta_vec) - cost(X,Y, new_theta)) <= convergence_criteria
        ):
            break
        i += 1
        theta_vec = new_theta
    return theta_vec, j_array


''' 
    H = theta_0 * 1 + theta_1 X1 + theta_2 X2
'''
#x0 x1 x2 where x0 is always 1
X = np.array([ 
    [1, 0, 1], 
    [1, 1, 0],
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
])
Y = np.array([0.05, 2.05, 1.05, 1.95, -0.05])

theta_vec = np.array([0,1,1])
learning_rate = 0.001
convergence_criteria = 0.00000001
max_iterations = 10000

estimated_theta, j_array = train_model(X,Y, theta_vec[:], learning_rate, convergence_criteria, max_iterations)
_, j_array_a_01 = train_model(X,Y, theta_vec[:], 0.01, convergence_criteria, max_iterations)
_, j_array_a_005 = train_model(X,Y, theta_vec[:], 0.005, convergence_criteria, max_iterations)

x1 = np.array([x[1] for x in X])
x2 =  np.array([x[2] for x in X])

#plots
fig = plt.figure()

#INPUT AND REGRESSION LINE PLOT
ax_input_data = fig.add_subplot(211, projection='3d')
ax_input_data.scatter(
    x1, x2, Y, color="r", label="Actual Data", marker="o"
)

predicte_data = [estimated_theta[0] + estimated_theta[1]*x[1] + estimated_theta[2]*x[2] for x in X]
ax_input_data.plot(
    x1, x2, predicte_data, color="g", label="Predicted Data"
)
ax_input_data.set(xlabel="X1", ylabel="X2", zlabel="Y", title="Input data and predicted data plot")
ax_input_data.legend()


# COST PLOT W.R.T ITERATIONS
ax_j_data = fig.add_subplot(212)
ax_j_data.set(xlabel="Iterations", ylabel="Cost", title="Cost vs Iteration graph")
ax_j_data.plot(j_array, label="alpha=0.001", linestyle="-")
ax_j_data.plot(j_array_a_01, label="alpha=0.01", linestyle="-.")
ax_j_data.plot(j_array_a_005, label="alpha=0.005", linestyle=":")
ax_j_data.legend()

plt.show()