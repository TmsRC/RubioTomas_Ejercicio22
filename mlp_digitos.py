import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection


numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)

scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


neuronas = np.linspace(1,20,20,dtype=int)
loss = []
f1_test = []
f1_train = []
it = 2500

for i in neuronas:
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic',hidden_layer_sizes=(i),max_iter=it)
    mlp.fit(X_train, Y_train)
    
    loss.append(mlp.loss_)
    f1_test.append(sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro'))
    f1_train.append(sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), average='macro'))
    print(i)
    
    
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.scatter(neuronas,loss,c='b',label='Loss')
ax2.scatter(neuronas,f1_test,c='r',label='Test')
ax2.scatter(neuronas,f1_train,c='b',label='Train')
ax1.set_title('Loss')
ax2.set_title('F1 Score')
ax1.set_xlabel('Número de neuronas')
ax2.set_xlabel('Número de neuronas')
ax1.set_xticks(neuronas[1::2])
ax2.set_xticks(neuronas[1::2])
ax2.legend()
fig.savefig('loss_f1.png')



best = 5
mlp = sklearn.neural_network.MLPClassifier(activation='logistic',hidden_layer_sizes=(best),max_iter=it)
mlp.fit(X_train, Y_train)


fig,axes = plt.subplots(1,best,figsize=(25,5))
for i in range(best):
    axes[i].imshow(mlp.coefs_[0][:,i].reshape(8,8))
    axes[i].set_title('Neurona {}'.format(i+1))

plt.savefig('neuronas.png')