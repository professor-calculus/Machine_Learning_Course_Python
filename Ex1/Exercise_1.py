##!/usr/bin/env python
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time

#----------------------------------
print('Linear regression: Population of city vs profit of food truck:')
df = pd.read_csv('../machine-learning-ex1/ex1/ex1data1.txt', names=['population', 'profit'])
df.head()

#plt.ion()
X = np.stack( (np.ones(len(df)), np.array(df['population'])), axis=-1)
print(X)

Y = np.array(df['profit'])
print(Y)

#Hypothesis h_theta(x)
def h(X, theta):
    h = np.inner(X, theta)
    return h;

def show_plot(theta):
    #plt.clf()
    #plt.cla()
    #plt.close()
    f,a = plt.subplots()
    plt.scatter(df['population'], df['profit'])
    x1 = np.linspace(0.0, 25.0, 100)
    X_model = np.stack( (np.ones(100), x1), axis=-1)
    y_model = h(X_model, theta)
    a.plot(x1, y_model, c='r')
    #plt.pause(0.001)

#show_plot([0., 1.])

def cost(x, y, theta):
    m = len(y)
    cost = np.sum( (h( x, theta) - y)**2 )
    cost *= 1.0/(2.0*m)
    return cost

#print(cost(X, Y, [-5.0, 1.2]))

def gradient_descent(X, y, alpha, theta):
    m = len(y)
    gradient = np.inner(np.transpose(X), (h(X,theta) - y))/m
    theta_upd = theta - alpha * gradient
    return theta_upd


print('Testing model')
theta = [4.0, -2.]
alpha = 0.01
for it in range(500):
    theta = gradient_descent(X, Y, alpha, theta)
    #show_plot(theta)
    C = cost(X, Y, theta)
    print("Iteration: {}".format(it))
    print("Cost: {}".format(C))
    print("Theta: {}".format(theta))
#plt.ioff()
show_plot(theta)
plt.show()

plt.clf()
plt.cla()

theta_x = np.linspace(-10., 10., 100)
theta_y = np.linspace(-1., 4., 100)
meshx, meshy = np.meshgrid(theta_x, theta_y)
costs = np.zeros((100,100))

for i in range(len(theta_x)):
    for j in range(len(theta_y)):
        costs[i][j] = cost(X, Y, [theta_x[i],theta_y[j]])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(meshx, meshy, costs, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()