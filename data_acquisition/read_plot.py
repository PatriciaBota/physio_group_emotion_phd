#Avaliação de quais as melhores configurações para a comparação entre o Bitalino e o Riot
'''
Lista das soluções testadas:
    Reconfigurar o Riot para usar o IP fixo utilizando a router riot
    Alterar a sampling frequency do bitalino de 1000 Hz para 100 Hz
    Alterar a sampling frequency do Riot de 100 Hz para 62.5 Hz (Alterei a Sample Rate na página de configurações do Riot de 10 para 16)
    Alterar a sampling frequency do Riot de 62.5 Hz para 40 Hz (Alterei a Sample Rate na página de configurações do Riot de 16 para 25)

Penso que esta descrição das soluções esteja mais ou menos completa uma vez que fiquei com algumas questões conseptuais de como é que estas alterações iam afetar o sistema em si. 

Para casa teste, recolhi cerca de 10 minutos de dados e realizei alguns plot para avaliar os resultados. Chamava a atenção para os graficos com o numero de pacote no eixo do y, nos 4 primeiros testes gráficos tem 2 declives, inicialmente os valores vão aumentando de 1 em 1 como é suposto mas apartir de um certo ponto começam a aumentar bastante, com diferenças significativas entre si, ou seja, elevado numero de pacotes perdidos (o nº de pacotes perdidos chega a atingir valores de 60/70). No entanto, isto não acontece com a ultima solução, neste caso, o gráfico com o numero de pacote só tem um declive, e embora por vezes haja perdas de pacotes a diferença entre 2 numeros de pacotes consecutivos não ultrapassa os 18.

Os graficos do PPG, após uma avaliação visual parece estar de acordo com o esperado. 

Com base nos resultados do Rafael da Sampling freq. minima para o PPG e no teorema de Nquist (assumindo uma freq. do sinal na casa do 15 Hz), penso que a ultima solução com 40 Hz de sampling freq. vá funcionar.
'''
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pathlib

#%%

list_files = ["2021419_111732_DATA.hdf5"] #list(pathlib.Path('DATA/').glob('*.hdf5'))
for filename in list_files:
  
    f = h5py.File(filename, "r")

    group = f['17']
    dataR=group['EDA'][()]

    #group = f['20B']
    #dataB=group['EDA'][()]

    td = 0
    prev=dataR[:,12][0]
    lostpackages=np.array([])
    for i in dataR[:,12][1:-1][np.argsort(dataR[1:-1,-1])]:
        if i-prev>1:
            lostpackages=np.append(lostpackages,(i-prev)-1)
            td += (i-prev)-1            
        elif i - prev <0:
            nl = (i-0) + (2**16-1 - prev)
            lostpackages=np.append(lostpackages,nl)
            td += nl 
        td +=1
        
        prev=i
        
    #df = np.diff(dataR[:,12]).astype(int)
    #df = df.tolist()
    
    #print("0", np.unique(df))    
    #df.remove(1)
    #df = list(filter(lambda a: a != 1, df))
    
    #print("1", np.unique(df))    
    
    #df.remove(-2**16+1)
    #df = list(filter(lambda a: a != -2**16+1, df))
    
    #print("2", np.unique(df))    
    #df = np.array(df)-1
    print('-----------------')     
    #print(name)
    print(f'Total lost packages:{sum(lostpackages)}')
    print(f'Nº of occurances w/ loss of packages:{len(lostpackages)}')
    print(f'Nº of received packages:{len(dataR[:,12])}')
    print(f'% of loss packages:{sum(lostpackages)*100/((dataR[:,-1][-1]-dataR[:,-1][0])*200)}')
    print(f'% of loss packages2:{sum(lostpackages)*100/td}')
    #print(f'% of loss packages 2:{sum(df)*100/((dataR[:,-1][-1]-dataR[:,-1][0])*200)}')
    print(f'Last package number: {dataR[:,12][-1]}')
    print(f'First package number: {dataR[:,12][0]}')
    a
    plt.figure()
    plt.plot(dataB[:,5][5000:5500])
    plt.savefig(f'Graficos\{name}_Bitalino_Channel_5_(PPG)_Zoom.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.ylabel('Amplitude')
    #plt.show()

    plt.figure()
    plt.plot(dataB[:,5])
    plt.savefig(f'Graficos\{name}_Bitalino_Channel_5_(PPG).png', dpi=300, transparent=False, bbox_inches='tight')
    plt.ylabel('Amplitude')
    #plt.show()

    plt.figure()
    plt.plot(dataR[:,12])
    plt.ylabel('Package number')
    plt.savefig(f'Graficos\{name}_Riot_Package_number_plot.png', dpi=300, transparent=False, bbox_inches='tight')
    #plt.show()

    plt.figure()
    plt.plot(lostpackages)
    plt.ylabel('Lost Packages')
    plt.savefig(f'Graficos\{name}_Riot_Difference_between_consecutive_package_number.png', dpi=300, transparent=False, bbox_inches='tight')
    #plt.show()

    plt.figure()
    plt.plot(dataR[:,13])
    plt.ylabel('Time')
    plt.savefig(f'Graficos\{name}_Riot_Time_plot.png', dpi=300, transparent=False, bbox_inches='tight')
    #plt.show()



# %%
'''
name='A'
filename='2021419_111732_DATA.hdf5'
f=h5py.File(filename, "r")


group = f['17']
dataR=group['EDA'][()]


prev=dataR[:,12][0]
lostpackages=np.array([])
for i in dataR[:,12][1:-1]:
    if i-prev>1 and i!=0:
        lostpackages=np.append(lostpackages,(i-prev)-1)
    prev=i

         
print(name)
print(sum(lostpackages))
print(len(lostpackages))
print(len(dataR[:,12]))
print(f'Last package number: {dataR[:,12][-1]}')
print(f'First package number: {dataR[:,12][0]}')


plt.figure()
plt.plot(dataR[:,12])
plt.ylabel('Package number')
plt.savefig(f'Graficos\{name}_Riot_Package_number_plot.png', dpi=300, transparent=False, bbox_inches='tight')
#plt.show()

plt.figure()
plt.plot(lostpackages)
plt.ylabel('Lost Packages')
plt.savefig(f'Graficos\{name}_Riot_Difference_between_consecutive_package_number.png', dpi=300, transparent=False, bbox_inches='tight')
#plt.show()

plt.figure()
plt.plot(dataR[:,13])
plt.ylabel('Time')
plt.savefig(f'Graficos\{name}_Riot_Time_plot.png', dpi=300, transparent=False, bbox_inches='tight')
#plt.show()
# %%
'''
