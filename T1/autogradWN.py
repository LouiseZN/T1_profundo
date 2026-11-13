import numpy as np

class Tensor:
    def __init__(self, dimencao, filhos= ()):
        self.peso = np.zeros(dimencao)
        self.gradiente = np.zeros(dimencao,float)
        self.regra = lambda: None # Mudança ja que agora regra não irá receber um valor
        self.anteriores = set(filhos)  # Adicionamos isso para poder calcular a ordem topologic
    
    # As duas property abaixo servem para garantir que peso sempre seja um float, isso é necessario para evitar o problema de que o numpy não permite que um inteiro seja elevado a uma potência negativa
    @property
    def peso(self):
        return self._peso

    @peso.setter
    def peso(self, valor):
        self._peso = np.asarray(valor, dtype=float)
        
        
    # Função responsavel por permitir o backward pass mesmo sob broadcast
    @staticmethod
    def soma_dimensional(gradiente, shape):
        while gradiente.ndim > len(shape): # Isso elimina dimensões extras adicionadas via boradcasting pelo numpy
            gradiente = gradiente.sum(axis=0)
            
        for i, dim in enumerate(shape): # Isso reduz a dimensionalidade de dimensões existentes de volta para 1 (caso originalmente fosse um)
            if dim == 1 and gradiente.shape[i] != 1:
                gradiente = gradiente.sum(axis=i, keepdims=True) # keepdims = True mantem a dimensão original do tensor, na pratica ele adiciona um parenteses e mantem como dimensão de tamanho 1
                
        return gradiente
            
        

    def __str__(self):
        return str(self.peso) # se for chamada vai retornar so o peso 

    def __add__(self, outra):
        if isinstance(outra, (int, float)):
            a = Tensor([1])
            a.peso = np.array([outra])
            outra = a
        
        resultado = self.peso + outra.peso
        retorno = Tensor(resultado.shape, (self, outra))
        retorno.peso = resultado

        def regra():
            self.gradiente += self.soma_dimensional(retorno.gradiente, self.peso.shape)
            outra.gradiente += self.soma_dimensional(retorno.gradiente, outra.peso.shape)
            
        retorno.regra = regra
        
        return retorno
    

    def __mul__(self, outra):  # ponto a ponto 
        if isinstance(outra, (int, float)):
            a = Tensor([1])
            a.peso = np.array([outra])
            outra = a
            
        
        resultado = self.peso * outra.peso
        retorno = Tensor(resultado.shape, (self, outra))
        retorno.peso = resultado

        def regra():
            self.gradiente += self.soma_dimensional(retorno.gradiente * outra.peso, self.peso.shape) # += pq soma os dois gradientes p um tensor com 2 saidades vai alterar x + y no final (por isso somamos)
            outra.gradiente += self.soma_dimensional(retorno.gradiente * self.peso, outra.peso.shape) 
        
        retorno.regra = regra
 
        return retorno
    
    def __pow__(self, outra): # Exponenciação (outra aqui é um escalar). Usaremos isso para fazer a divisão apartir da multiplicação
            retorno  = Tensor(self.peso.shape, (self,))
            retorno.peso = self.peso ** outra
            
            def regra():
                self.gradiente += outra * (self.peso ** (outra-1)) * retorno.gradiente  
                
            retorno.regra = regra
            
            return retorno   
    
    def __rmul__(self, outra): # Permitir multiplicação por escalar com o escalar do lado direito
        return self.__mul__(outra)
    
    def __radd__(self, outra): # Permitir soma por escalar com o escalar do lado direito
        return self.__add__(outra)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, outra):
        return self.__add__(-outra)
    
    def __rsub__(self, outra):
        return outra + (-self)   
        
    def __truediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self * (1 / other)
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self **-1) * other
    
    def __matmul__(self, outra):
        a = self.peso @ outra.peso
        retorno = Tensor(a.shape, (self, outra))
        retorno.peso = a
        
        def regra():
            self.gradiente +=retorno.gradiente @ outra.peso.T
            outra.gradiente += self.peso.T @ retorno.gradiente
            
        retorno.regra = regra
        
        return retorno
    
    def log(self, base="euler"):
        if base == "euler":
            resultado = np.log(self.peso)
            retorno = Tensor(resultado.shape, (self,))
            retorno.peso = resultado
            
            def regra():
                self.gradiente += retorno.gradiente/self.peso
                
            retorno.regra = regra
            
            return retorno
        
        resultado = np.log(self.peso) / np.log(base) # Propriedade de mudança de base dos logaritmos
        retorno = Tensor(resultado.shape, (self,))
        retorno.peso = resultado

        def regra():
            self.gradiente += retorno.gradiente / (self.peso * np.log(base))

        retorno.regra = regra

        return retorno

    def root(self, value = 2): # Podemos aproveitar a implementação em potencia para fazer a radicialização
            return self ** (1/value)
    
    def sum(self, axis=None):
        resultado = np.sum(self.peso, axis=axis)
        retorno = Tensor(resultado.shape, (self,))
        retorno.peso = resultado
        
        def regra():
            
            gradiente = retorno.gradiente
            
            if axis is not None: # Isso é para impedir o bug de caso o resultado final do sum no foarward tenha sido um escalar
                # Expandindo a dimensão removida durante o foarward
                gradiente = np.expand_dims(gradiente, axis=axis)
            
            self.gradiente += np.broadcast_to(gradiente, self.peso.shape) # Copia as dimensões em 1 para bater com o valor da matriz
            
            
        retorno.regra = regra
        
        return retorno
    
    
    
    # Funções de ativação

    def reLU(self):
        retorno = Tensor(self.peso.shape, (self,))
        retorno.peso = np.where(self.peso>0, self.peso, 0)

        def regra():
            self.gradiente += (self.peso > 0) * retorno.gradiente

        retorno.regra = regra

        return retorno
    
    
    def sigmoid(self):
        resultado = 1 / (1 + np.exp(-self.peso))

        retorno = Tensor(resultado.shape, (self,))
        retorno.peso = resultado

        def regra():
            self.gradiente += retorno.gradiente * retorno.peso * (1 - retorno.peso)

        retorno.regra = regra

        return retorno
    
    
    def tanh(self):
        resultado = np.tanh(self.peso)
        retorno = Tensor(resultado.shape, (self,))
        retorno.peso = resultado
        
        def regra():
            self.gradiente += (1 - resultado**2) * retorno.gradiente
            
        retorno.regra = regra
        
        return retorno
    
    def linear(self): # Essa aqui foi difícil 
        return self
    
    def leakyReLU(self, alpha=0.01):
        resultado = np.where(self.peso>0, self.peso, alpha*self.peso)
        retorno = Tensor(self.peso.shape, (self,))
        retorno.peso = resultado
        
        def regra():
            self.gradiente += np.where(self.peso>0, 1, alpha) * retorno.gradiente
            
        retorno.regra = regra
        
        return retorno

            
    def backward(self): # calulo dos gradientes
        self.gradiente = np.ones_like(self.peso, dtype=float)
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.anteriores:
                    build_topo(child)
                topo.append(v) # Note que quando chamamos a função com self, ele será o ultimo a ser visitado ja que é um dfs
        build_topo(self)
        
        for nodo in reversed(topo):
            nodo.regra()
            
    def __call__(self, outro):
        return self @ outro



def Dense(input, output, inicializacao = "He", limite_inferior=0, limite_superior=1, desvio_padrao=1, media=0):
    matriz = Tensor([input, output])
    
    if inicializacao == "He":
        matriz.peso = np.random.normal(0,
                                       np.sqrt( 2/input),
                                       size=(input,output))
    elif inicializacao == "Xavier":
        matriz.peso = np.random.normal(0,
                                       np.sqrt( 2/(input+output)),
                                       size=(input,output))
                                       
    elif inicializacao == "Uniforme":
        matriz.peso = np.random.uniform(limite_inferior, limite_superior, size=(input, output))
    elif inicializacao == "Normal":
        matriz.peso = np.random.normal(media,
                                       np.sqrt(desvio_padrao), 
                                       size=(input, output))
        
    return matriz


class MLP:
    def __init__(self, *camadas):
        self.camadas = camadas


    def __call__(self, x):
        for pesos, ativacao in self.camadas:
            x = x @ pesos
            x = getattr(x, ativacao)()

        return x
    
    def parametros(self):
        return [pesos for pesos, _ in self.camadas]
    
    def zerar_gradiente(self):
        for pesos in self.parametros():
            pesos.gradiente = np.zeros_like(pesos.gradiente)
            
            






def MSE(y, y_pred):
    custo = ((y - y_pred)**2).sum()
    return custo / y.peso.size

def BCE(y, y_pred):
    custo = -(y * y_pred.log() + (1-y) * (1-y_pred).log())
    return custo.sum() / y.peso.size

