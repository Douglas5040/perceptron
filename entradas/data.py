import numpy as np

def v(): return np.array([
                #nomal
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