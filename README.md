# Descrição da atividade:
Implementação de um algoritmo Perceptron com 1 neurônio artificial.  Tendo como objetivo classificar figuras com a letra "A" (cujo label é definido como +1) e a letra "∀" (cujo label é definido como -1).

Data: 10 de setembro de 2020
Turma: Redes Neurais Código: SCC 5809

# Equipe:
ID. Matricula (01) - 12116252 Dheniffer Caroline Araújo Pessoa

ID. Matricula (02) - 12114819 Douglas Queiroz Galucio Batista 

ID. Matricula (03) - 12116738 Laleska Mesquita

# Instruções:  
Implementar e treinar o modelo Adaline par reconhecer os símbolos A e A Invertido.
Faça uma representação matricial de -1 e +1 para desenhar esses símbolos graficamente, e crie vários exemplos de treinamento e teste, inserindo ruídos arbitrariamente. 

Desenvolvimento: O ambiente utilizado para a compilação do algoritmo foi o Visual Studio Code.

# Código referente as “Entradas”
import numpy as np

def v(): return np.array([

                #normal
 
                    #1
                    [[+1, -1, +1, -1, -1],
                    [-1, +1, -1, +1, -1],
                    [-1, +1, +1, +1, -1],
                    [-1, +1, +1, +1, -1],
                    [+1, +1, -1, +1, +1]],

                    #2
                    [[+1, -1, +1, -1, +1],
                    [-1, +1, -1, +1, -1],
                    [-1, +1, +1, +1, -1],
                    [+1, -1, -1, -1, +1],
                    [+1, -1, +1, -1, +1]],

                    #3
                    [[+1, -1, +1, -1, +1],
                    [-1, +1, -1, +1, -1],
                    [+1, +1, +1, +1, -1],
                    [-1, +1, -1, +1, -1],
                    [+1, +1, -1, +1, +1]],

                    #4
                    [[-1, +1, +1, -1, -1],
                    [+1, -1, -1, +1, -1],
                    [+1, +1, +1, +1, -1],
                    [+1, -1, -1, +1, -1],
                    [+1, -1, -1, +1, -1]],

                    #5
                    [[-1, -1, +1, +1, -1],
                    [-1, +1, -1, -1, +1],
                    [-1, +1, +1, +1, +1],
                    [-1, +1, -1, -1, +1],
                    [-1, +1, -1, -1, +1]],

 

 
                #invertido
                    #1
                    [[+1, -1, -1, -1, +1],
                    [+1, +1, +1, +1, +1],
                    [-1, +1, -1, +1, -1],
                    [-1, +1, -1, +1, -1],
                    [+1, -1, +1, -1, +1]],
                    
                    #2
                    [[+1, -1, -1, -1, +1],
                    [+1, +1, -1, +1, +1],
                    [-1, +1, +1, +1, -1],
                    [-1, +1, -1, +1, -1],
                    [+1, -1, +1, -1, +1]],
                    
                    #3
                    [[+1, -1, +1, -1, +1],
                    [+1, -1, -1, -1, +1],
                    [-1, +1, +1, +1, -1],
                    [+1, +1, -1, +1, -1],
                    [-1, -1, +1, -1, +1]],
                    

                    #4
                    [[+1, -1, +1, -1, +1],
                    [+1, -1, +1, -1, +1],
                    [-1, +1, +1, +1, -1],
                    [+1, +1, -1, +1, -1],
                    [+1, -1, +1, -1, +1]],
                    
                    #5
                    [[+1, -1, -1, +1, -1],
                    [+1, -1, -1, +1, -1],
                    [+1, +1, +1, +1, -1],
                    [+1, -1, -1, +1, -1],
                    [-1, +1, +1, -1, +1]]
                ])
#print (v())

# Código Referente ao “Treinamento” 

import numpy as np

def p(): return np.array([
            #normal
                #1
                [[-1, -1, +1, -1, -1],
                [-1, +1, -1, +1, -1],
                [-1, +1, +1, +1, -1],
                [+1, -1, -1, -1, +1],
                [+1, -1, -1, -1, +1]],

                #2
                [[-1, -1, +1, -1, -1],
                [-1, +1, -1, +1, -1],
                [-1, +1, +1, +1, -1],
                [-1, +1, -1, +1, -1],
                [-1, +1, -1, +1, 1]],

                #3
                [[-1, -1, +1, -1, -1],
                [-1, +1, -1, +1, -1],
                [-1, +1, +1, +1, -1],
                [+1, +1, +1, +1, +1],
                [+1, -1, -1, -1, +1]],

                #4
                [[-1, -1, +1, -1, -1],
                [-1, +1, -1, +1, -1],
                [-1, +1, +1, +1, -1],
                [-1, +1, -1, +1, -1],
                [+1, +1, -1, +1, +1]],

            #invertido
                #1
                [[+1, -1, -1, -1, +1],
                [+1, -1, -1, -1, +1],
                [-1, +1, +1, +1, -1],
                [-1, +1, -1, +1, -1],
                [-1, -1, +1, -1, -1]],
                
                #2
                [[+1, -1, -1, -1, +1],
                [+1, -1, -1, -1, +1],
                [-1, +1, +1, +1, -1],
                [-1, +1, -1, +1, -1],
                [+1, -1, +1, -1, +1]],
                
                #3
                [[-1, +1, -1, +1, -1],
                [-1, +1, -1, +1, -1],
                [-1, +1, +1, +1, -1],
                [-1, +1, -1, +1, -1],
                [-1, -1, +1, -1, +1]],
                
                #4
                [[+1, -1, -1, -1, +1],
                [+1, +1, +1, +1, +1],
                [-1, +1, +1, +1, -1],
                [-1, +1, -1, +1, -1],
                [-1, -1, +1, -1, +1]]
            ])
#print (p())

# Código Referente ao Algoritmo de “Perceptron”

import numpy as np

import entradas.data as entra_A_data
import treinamento.data as trei_A_data

#print('a_normal', a_normal.p())

#p: Vetor com as matrizes de entrada.

p = trei_A_data.p()

#p_r: vetor com o valor de cada matriz em p, em ordem

p_r = np.array([1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0])

#w: pesos da rede, um peso pra dada posição da matriz

w = np.zeros((5,5))

# BIAS
b = np.zeros(1)


# Função de ativação
def f_ativacao(x):
    if x <= 0.0:
        return -1.0
    return 1.0

alpha = 0.5
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

# Avaliação

#v: matriz para avaliação

v = entra_A_data.v()

#v_r: vetor com o valor de cada matriz em v, em ordem

v_r = np.array([1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0])

#print(v_r)

for i_v in range(len(v)):
    
    print(str(i_v+1)+" - Iteração Matriz ("+str(v_r[i_v])+"):",perceptron(v[i_v]))
