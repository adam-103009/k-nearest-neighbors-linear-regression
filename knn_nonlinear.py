import numpy as np
import matplotlib.pyplot as plt
# Knn Regression 
class myKnnLinearRegression():
    def __init__(self,k):
        self.n_neighbors = k
    def fit(self,x,y):
        self.train_X = x
        self.train_Y = y
    def predict(self,test_point):
        # 計算test到所有點的距離，並建立table存取
        distance_table = self.calculate_distance(test_point)
        num_test = test_point.shape[0]
        predict_y = np.zeros([num_test])
        for i in range(num_test):
            # 尋找最近的k個點的index
            neighbors = self.search_closet_point(distance_table,i)
            # 根據index取出點的x,y值
            neighbors_x, neighbors_y = self.getNeighbors(neighbors)
            predict_y[i] = neighbors_y.mean()
        return predict_y
    def search_closet_point(self,distance_table,i):
        neighbors = np.argsort(distance_table[i])[:self.n_neighbors]
        return neighbors
    def getNeighbors(self,neighbors):
        neighbors_x = np.zeros([self.n_neighbors,self.train_X.shape[1]])
        neighbors_y = np.zeros([self.n_neighbors])
        for i in range(self.n_neighbors):
            neighbors_x[i] = self.train_X[neighbors[i]]
            neighbors_y[i] = self.train_Y[neighbors[i]]
        return neighbors_x,neighbors_y  
    def calculate_distance(self,test_point):
        num_test = test_point.shape[0]
        num_train = self.train_X.shape[0]
        every_point_distance = np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                every_point_distance[i][j] = np.sum((test_point[i]-self.train_X[j])**2)**0.5
        return every_point_distance
    def linear_regression(self,x,y):
        x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        y=y[:,np.newaxis]
        beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
        return beta
def plot_scatter(x,y,predict_y):
    if(x.shape[1]==1):
        plt.scatter(x,y)
        plt.scatter(x,predict_y,c='r')
        plt.show()
    else:
        x1 = []
        x2 = []
        for j in range(len(x)):
            x1.append(x[j][0])
            x2.append(x[j][1])
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x1,x2,y)
        ax1.scatter(x1,x2,predict_y)
        ax1.set_xlabel('x1 axis')
        ax1.set_ylabel('x2 axis')
        ax1.set_zlabel('y axis')
        plt.show()
def getRMSE(truth_value,predict_value):
    out = 0
    for i in range(len(predict_value)):
        out = out+(predict_value[i]-truth_value[i])**2
    out = (out/len(predict_value))**0.5
    return out
def splictTrainTest(data_x,data_y,train_size):
    if(len(data_x.shape) == 1):
        train_X = []
        for i in range(train_size):
            train_X.append(np.array([data_x[i]]))
        test_x = []
        for i in range(train_size,len(data_x),1):
            test_x.append(np.array([data_x[i]]))
        train_X = np.array(train_X)
        test_x = np.array(test_x)
    else:
        train_X = np.array(data_x[:train_size])
        test_x = np.array(data_x[train_size:])
    train_Y = np.array(data_y[:train_size])
    test_y = np.array(data_y[train_size:])
    return train_X,train_Y,test_x,test_y
if __name__ == '__main__':
    # 讀取資料 data1
    data = np.load('data/data1.npz')
    train_size = 700
    k = 7

    model = myKnnLinearRegression(k = k)
    train_X,train_Y,test_x,test_y = splictTrainTest(data['X'],data['y'],train_size)
    model.fit(train_X,train_Y)
    predict_y = model.predict(test_x)
    # calculate root mean square error
    rmse = getRMSE(test_y,predict_y)
    print("RMSE for data1",rmse)
    data1_test_x,data1_test_y,data1_predict_y = test_x,test_y,predict_y
    #plot_scatter(test_x,test_y,predict_y)

    # 讀取資料 data2
    data = np.load('data/data2.npz')
    train_size = 700
    model = myKnnLinearRegression(k = k)
    train_X,train_Y,test_x,test_y = splictTrainTest(data['X'],data['y'],train_size)
    model.fit(train_X,train_Y)
    predict_y = model.predict(test_x)
    # calculate root mean square error
    rmse = getRMSE(test_y,predict_y)
    print("RMSE for data2",rmse)
    
    # 畫圖
    #data1
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.set_title("data1")
    ax.scatter(data1_test_x,data1_test_y)
    ax.scatter(data1_test_x,data1_predict_y,c='r')
    #data2

    x1 = []
    x2 = []
    for j in range(len(test_x)):
        x1.append(test_x[j][0])
        x2.append(test_x[j][1])
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.set_title("data2")
    ax.scatter(x1,x2,test_y)
    ax.scatter(x1,x2,predict_y)
    ax.set_xlabel('x1 axis')
    ax.set_ylabel('x2 axis')
    ax.set_zlabel('y axis')

    plt.show()
    #plot_scatter(test_x,test_y,predict_y)