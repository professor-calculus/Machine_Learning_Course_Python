##!/usr/bin/env python
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
matplotlib.rcParams['text.usetex'] = True
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
    plt.xlabel('Population (10,000s)')
    plt.ylabel('Profit (\$10,000s)')
    x1 = np.linspace(0.0, 25.0, 100)
    X_model = np.stack( (np.ones(100), x1), axis=-1)
    y_model = h(X_model, theta)
    a.plot(x1, y_model, c='r')
    #plt.pause(0.001)

def show_plot_multi(theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( (df['size']*std_size + mean_size), (df['bedrooms']*std_bedrooms + mean_bedrooms), df['price'])
    ax.set_xlabel('House Size (square feet)')
    ax.set_ylabel('$\#$ of Bedrooms')
    ax.set_zlabel('Price (\$)')
    #x1 = np.linspace(-4., 4., 100)
    x1 = np.linspace(0., 4000., 100)
    x2 = np.linspace(0., 6., 100)
    X_model = np.stack( (np.ones(100), (x1-mean_size)/std_size, (x2-mean_bedrooms)/std_bedrooms), axis=-1)
    y_model = h(X_model, theta)
    print(h([1,(1650.-mean_size)/std_size,(3.-mean_bedrooms)/std_bedrooms], theta))
    ax.plot(x1, x2, y_model, c='r')
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
ax.set_xlabel('$\\theta_{0}$')
ax.set_ylabel('$\\theta_{1}$')
plt.show()

fig = plt.figure()
levels = np.logspace(-1, 3, 30)
CS = plt.contour(meshx, meshy, costs, levels)
plt.xlabel('$\\theta_{0}$')
plt.ylabel('$\\theta_{1}$')
plt.show()

print('\n -------------------- \n')

#-----------------------------
print('Linear regression with multiple variables')
df = pd.read_csv('../machine-learning-ex1/ex1/ex1data2.txt', names=['size', 'bedrooms', 'price'])
print(df)

mean_size = df['size'].mean()
std_size = df['size'].std()
mean_bedrooms = df['bedrooms'].mean()
std_bedrooms = df['bedrooms'].std()

df['size'] = (df['size'] - mean_size)/std_size
df['bedrooms'] = (df['bedrooms'] - mean_bedrooms)/std_bedrooms
print(df)

X = np.stack( (np.ones(len(df)), np.array(df['size']), np.array(df['bedrooms'])), axis=-1)
print(X)

Y = np.array(df['price'])

print('Testing model')
theta = [300000., 200000., 100]
alpha = 0.01
for it in range(5000):
    theta = gradient_descent(X, Y, alpha, theta)
    #show_plot(theta)
    C = cost(X, Y, theta)
    print("Iteration: {}".format(it))
    print("Cost: {}".format(C))
    print("Theta: {}".format(theta))
#plt.ioff()
show_plot_multi(theta)
plt.show()