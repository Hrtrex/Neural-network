import numpy as np
import matplotlib.pyplot as plt

def rands(s, p):
    # Funkcja inicjalizująca wagi oraz przesunięcia neuroów
    # s - liczba neuronów w warstwie
    # p - macierz danych wejściowych
    # w, b - zwraca całkowicie losowe wartości 
    # macierzy wag i wektora przesunięć
    w = 2 * np.random.rand(np.size(s), 1) - 1
    b = 2 * np.random.rand(np.size(p), 1) - 1
    #w = 2 * np.random.rand() - 1
    #b = 2 * np.random.rand() - 1
    #return np.array([[w]]), np.array([[b]]) 
    return w, b

def nwlog(s, p):
    # Funkcja inicjalizująca wagi oraz przesunięcia neuronów
    # za pomocą metody Nguyen-Widrow'a dla neuronów logistycznych
    # s - liczba neuronów w warstwie
    # p - macierz danych wejściowych
    # w, b - zwraca macierz wag i wektor przesunięć
    p = np.ones((p,1)) * np.array([0, 1])
    r = np.size(np.size(p))
    pmin = np.min(p)
    pmax = np.max(p)
    magw = 2.8 * s ** (1/r)
    w = magw * np.random.uniform(-1, 1, (s, r))
    b = magw * np.random.uniform(-1, 1, (s, r))
    rng = pmax - pmin
    mid = 0.5 * (pmin+pmax)
    w = 2 * w / (np.ones((s, 1)) * rng)
    b = b - w * mid
    return w, b

def nwtan(s, p):
    # Funkcja inicjalizująca wagi oraz przesunięcia neuronów
    # za pomocą metody Nguyen-Widrow'a dla neuronów tangensa-hiperbolicznego
    # s - liczba neuronów w warstwie
    # p - macierz danych wejściowych
    # w, b - zwraca macierz wag i wektor przesunięć
    p = np.ones((p,1)) * np.array([-1, 1])
    r = np.size(np.size(p))
    pmin = np.min(p)
    pmax = np.max(p)
    magw = 0.7 * s ** (1/r)
    w = magw * np.random.uniform(-1, 1, (s, r))
    b = magw * np.random.uniform(-1, 1, (s, r))
    rng = pmax - pmin
    mid = 0.5 * (pmin+pmax)
    w = 2 * w / (np.ones((s, 1)) * rng)
    b = b - w * mid
    return w, b

def purelin(n, *b):
    # Funkcja liniowa y = n
    if not b:
        return n
    else:
        return (n + b * np.ones((np.size(n),1)))[0][0]

def logsig(n, *b):
    # Sigmoidalna funkcja unipolarna
    if not b:
        return 1 / (1 + np.exp(-n))
    else:
        return 1 / (1 + np.exp(-(np.array(n) + b)))[0][0]

def tansig(n, *b):
    # Sigmoidalna funkcja bipolarna
    if not b:
        return np.tanh(np.array(n))
    else:
        return np.tanh(np.array(n)+b)[0][0]

def deltatan(a, e):
    # Funkcja obliczająca deltę dla neuronów tangensa-hiperbolicznego
    # a - macierz wektorów wyjściowych
    # d - macierz błędów
    return (1 - a * a) * e

def deltalog(a, e):
    # Funkcja obliczająca deltę dla neuronów logarytmicznych
    # a - macierz wektorów wyjściowych
    # d - macierz błędów
    return a * (1 - a) * e

def deltalin(a, e):
    # Funkcja obliczająca deltę dla neuronów liniowych
    # a - macierz wektorów wyjściowych
    # d - macierz błędów
    #return np.array(e)
    return e

def learnbp(p, d, lr):
    # Funkcja uczenia metodą propagacji wstecznej
    # p - macierz wektorów wejściowych
    # d - macierz wektorów błędów
    # lr - współczynnik uczenia
    dw = np.dot(lr * d, p)
    db = np.dot(lr * d, np.ones((np.size(p), 1)))
    return dw, db

class NeuralNetwork:
    def __init__(self, P, T, activation='tansig'):
        # Klasa realizująca uczenie sieci jednowarstwowej
        # P - wektor danych wejściowych
        # T - wektor danych docelowych
        # activation - funkcja przejścia (domyślnie 'tansig')
        self.P = P
        self.T = T
        self.R = np.size(np.size(P))
        self.S1 = np.size(np.size(T))
        self.D1 = None
        # przypisanie odpowiednich funkcji na podstawie 
        # podanej funkcji przejścia
        if activation == 'tansig':
            self.activation = tansig
            self.train = deltatan
            self.initial = nwtan
        elif activation == 'logsig':
            self.activation = logsig
            self.train = deltalog
            self.initial = nwlog
        elif activation == 'purelin':
            self.activation = purelin
            self.train = deltalin
            self.initial = rands
        # wywołanie funkcji inicjalizacyjnej
        self.W1, self.B1 = self.initial(self.S1, self.R)
        # wywołanie funkcji aktywacji
        self.A1 = self.activation(self.W1*self.P, self.B1)
        # obliczenie wektora błędów 
        self.E = self.T - self.A1
        # obliczenie błędu sieci
        self.SSE = np.sum(self.E**2)
        # przygotowanie wykresów oraz
        # narysowanie docelowego wyjścia
        plt.ion()
        fig, ax = plt.subplots()
        self.line1, = ax.plot(self.P, self.A1) 
        plt.plot(P, T, 'r')
        ax.legend(['wyjście otrzymane', 'wyjście docelowe'])
        plt.title(f'{self.train.__name__}')
        plt.axis([-1, -.5, -1, 1])

    def learn(self, lr=0.1, epochs=20000, err_goal=0.01, disp_freq=10):
        # funkcja ucząca
        for epoch in range(0, epochs):
            if self.SSE < err_goal:
                epoch -= 1
                break
            # wywołanie funkcji uczących
            self.D1 = self.train(self.A1, self.E)
            dW1, dB1 = learnbp(self.P, self.D1, lr)
            # aktualizacja wag i przesunięć
            self.W1 += dW1
            self.B1 += dB1
            ###
            self.A1 = self.activation(self.W1*self.P, self.B1)
            self.E = self.T - self.A1
            self.SSE = np.sum(self.E**2)
            if epoch % disp_freq == 0:
                self.display_results(epoch, self.SSE)

    def display_results(self, epoch, SSE):
        # prezentacja wyników 
        # (aktualizowanie otrzymanego wyjścia)
        print(epoch, f'SSE: {SSE}')
        self.line1.set_ydata(self.A1)
        #plt.pause(1e-15)

if __name__ == '__main__':
    # wektor danych wejściowych
    P = np.array([-1, -.9, -.8, -.7, -.6, -.5])
    # wektor danych docelowych
    T = np.array([-.9602, -.5770, -.0729, .3771, .6405, .6600])
    # stworzenie obiektu klasy, podanie funkcji przejścia
    nn = NeuralNetwork(P, T, activation='tansig')
    #nn2 = NeuralNetwork(P, T, activation='logsig')
    #nn3 = NeuralNetwork(P, T, activation='purelin')
    # wywowałanie metody uczącej
    nn.learn(epochs=20001)
    #nn2.learn(epochs=20001)
    #nn3.learn(epochs=20001)
    # wstrzymanie wykonywania programu
    input()