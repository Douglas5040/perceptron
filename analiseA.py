import numpy as np
P = np.array([[+1, -1, -1, -1, +1],
              [+1, -1, -1, -1, +1],
              [-1, +1, +1, +1, -1],
              [-1, +1, -1, +1, -1],
              [-1, -1, +1, -1, +1]])

P2 =np.array([[+1, -1, -1, -1, +1],
              [+1, -1, +1, -1, +1],
              [-1, +1, +1, +1, +1],
              [+1, +1, +1, +1, +1],
              [+1, +1, +1, +1, +1]])


p = P2.ravel() # Tamanho: 25 x 1
p = np.expand_dims(p, axis=1)
# Um neurônio W --> (25 x 1)
W = np.ones(shape=(25, 1))
# BIAS
b = np.ones(shape=(1,))
# Função de ativação
#    a = funcao(n)
#    Adaline usa uma função de ativação Linear, portanto:
#    a = n
total_iter = 10
alpha = 0.001
for i in range(total_iter):
    # Forward
    n = np.sum(np.transpose(W)*p) + b
    E = (-1 - n)
    # Backward
    W = W + 2*alpha*E*np.transpose(p)
    b = b + 2*alpha*E
print("Saída da rede neural (depois do treinamento): " + str(n))