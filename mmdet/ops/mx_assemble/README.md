corr        |   assem
T           |   dT
A   pA      |   dA  pdA
B   pB      |   B   pB
dT          |   T
dA          |   A
dB          | dB

assemble forward input: pB, T
                output: A
corr forward input:     pB, dT
            output:     dA


assemble backward1 intput:  pdA, pB  | pdA, T
                  output:  dT        | dB

corr forward2 input:        pA,  pB
             output:        T
assemble backward2 input:              pA, dT
                  output:              dB

