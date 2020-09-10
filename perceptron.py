import numpy as np

import entradas.data as entra_A_data
import treinamento.data as trei_A_data

#print('a_normal', a_normal.p())
# p: Vetor com as matrizes de entrada.
p = trei_A_data.p()
# p_r: vetor com o valor de cada matriz em p, em ordem
p_r = np.array([1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0])
# w: pesos da rede, um peso pra dada posição da matriz
w = np.zeros((5,5))
# BIAS
b = np.zeros(1)


# Função de ativação
def f_ativacao(x):
    if x <= 0.0:
        return -1.0
    return 1.0

alpha = 0.6
erro  = 1.0
while(erro != 0.0):
    for i_matriz in range(len(p)):
        erro = p_r[i_matriz] - f_ativacao(np.sum(w*p[i_matriz])+b)
        if (erro == 0.0):
            continue

        # Backward
        w = w + alpha*erro*p[i_matriz]
        b = b + alpha*erro
        break

def perceptron(x):
    return f_ativacao(np.sum(w*x)+b)

#avaliação
# v: matriz para avaliação
v = entra_A_data.v()
# v_r: vetor com o valor de cada matriz em v, em ordem
v_r = np.array([1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
#print(v_r)

for i_v in range(len(v)):
    
    print(str(i_v+1)+" - Iteração Matriz ("+str(v_r[i_v])+"):",perceptron(v[i_v]))