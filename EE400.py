#Thomas Johann Hillermmann Gomes 206624

import numpy as np
from scipy.optimize import newton

#Constantes
u = 3.986004418e5  
TOA = 60000  
v = 300  

#Dados dos satélites em formato de dicionário
satelites = {
    'satelite_1': {'a': 15300, 'e': 0.41, 'w': 60, 'i': 30, 'o': 0, 'dt': 4708.5603},
    'satelite_2': {'a': 16100, 'e': 0.342, 'w': 10, 'i': 30, 'o': 40, 'dt': 5082.6453},
    'satelite_3': {'a': 17800, 'e': 0.235, 'w': 30, 'i': 0, 'o': 40, 'dt': 5908.5511},
    'satelite_4': {'a': 16400, 'e': 0.3725, 'w': 60, 'i': 20, 'o': 40, 'dt': 5225.3666}
}

#Tempos de transmissão
TOT = {
    'tempo_1': 13581.1080927,
    'tempo_2': 19719.32768037,
    'tempo_3': 11757.73393255,
    'tempo_4': 20172.46081236,
}

#Calcular o TOF para cada satélite
TOF = [(TOA - TOT[f'tempo_{i+1}']) / 1000 for i in range(len(satelites))]

# Matrizes de rotação
def rotacao_z(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])

def rotacao_x(alpha):
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]])

#Função para calcular a posição no ECI
def posicao(sat):
    a, e, w, i, o, dt = sat['a'], sat['e'], np.radians(sat['w']), np.radians(sat['i']), np.radians(sat['o']), sat['dt']
    
    #Cálculo do período e anomalia média
    T = 2 * np.pi * np.sqrt(a**3 / u)
    M_e = 2 * np.pi * dt / T

    # Resolver a equação de Kepler
    def kepler(E, M_e, e):
        return E - e * np.sin(E) - M_e

    def dist_kepler(E, M_e, e):
        return 1 - e * np.cos(E)

    E = newton(func=kepler, fprime=dist_kepler, x0=np.pi, args=(M_e, e))

    #Coordenadas no sistema perifocal
    xk = a * (np.cos(E) - e)
    yk = a * np.sin(E) * np.sqrt(1 - e**2)
    pos_perifocal = np.array([xk, yk, 0])

    # Transformação para o sistema ECI
    R = rotacao_z(o) @ rotacao_x(i) @ rotacao_z(w)
    return R @ pos_perifocal

#Calcular a posição de cada satélite
lista_r = [posicao(sat) for sat in satelites.values()]

#Função de gradiente para o cálculo de posição
def gradiente(lista_r, r, TOF):
    gradient = np.zeros(3)
    for i, pos_sat in enumerate(lista_r):
        p = v * TOF[i]
        distancia_vetor = r - pos_sat
        modulo = np.linalg.norm(distancia_vetor)
        R = 1 - (p / modulo)
        gradient += R * distancia_vetor
    return gradient

#Chute inicial para a posição
r = np.array([-6420., -6432., 6325.])

#Otimização da posição
for i in range(800):
    G = gradiente(lista_r, r, TOF)
    r = r - 0.6 * G

print("Posição final estimada:", r)
