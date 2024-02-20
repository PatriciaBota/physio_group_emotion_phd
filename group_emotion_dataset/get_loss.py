import csv
from cv2 import repeat
import os
import h5py
import datetime
import os
import glob
import numpy as np
from operator import itemgetter
 

def order(a):
    div = [0]

    temp = [0]

    topop = []

    for i in range(len(a)):
        if a[i][-1] == 0.0:
            topop += [i]


    #print(topop)
    topop.reverse()

    for p in topop:
        a.pop(p)

    copy_a = a

    for i in range(len(a)-3):
        if (a[i+1][-1] - a[i][-1] < -4000000000) and (a[i+2][-1] - a[i][-1] < -4000000000) and (a[i+3][-1] - a[i][-1] < -4000000000):
            div += [i]
            temp += [a[i][-1]]
    

    #print(div)
    #print(temp)
    
    for j in range(len(div)-1):
        a[div[j]:div[j+1]] = sorted(a[div[j]:div[j+1]],key=itemgetter(-1))
    
    a[div[-1]+1:] = sorted(a[div[-1]+1:],key=itemgetter(-1))

    count = 0

    for i in range(len(a)):
        if a[i] != copy_a[i]:
            count += 1

    count_perc = count/len(a)*100
    #print('nr of samples: '+str(len(a)))
    #print('nr of elements out of order: '+str(count))
    print('percentage of elements out of order: '+str(count_perc))

    return np.array(a)


def getError(rows, fs):
    loss = 0.0
    repeated = 0.0
    longest_loss = 0

    #print('List of Losses bigger than 1s and Times when they occured:')

    for i in range(len(rows)-1):
        if float(rows[i+1][-2]) == float(rows[i][-2]) and float(rows[i+1][-1]) == float(rows[i][-1]): #some messages are repeated
                repeated += 1.0

        if (float(rows[i+1][-1]) - float(rows[i][-1])) > 10000:

            if int((float(rows[i+1][-1]) - float(rows[i][-1]))/1000000 * fs) - 1 > 0:
                loss += int((float(rows[i+1][-1]) - float(rows[i][-1]))/1000000 * fs) - 1
            
            #if int((float(rows[i+1][-1]) - float(rows[i][-1]))/1000000 * fs) - 1 > fs:
            #    print('Time: ', rows[i+1][-1])
            #    print('Loss: ', int((float(rows[i+1][-1]) - float(rows[i][-1]))/1000000 * fs)-1)


            if int((float(rows[i+1][-1]) - float(rows[i][-1]))/1000000 * fs) > longest_loss:
                longest_loss = int((float(rows[i+1][-1]) - float(rows[i][-1]))/1000000 * fs)-1
    #print('nr of messages: ', len(rows)/100)
    #print("nr of repeated msgs: ", repeated)
    #print("percentage of repeated msgs: ", repeated/len(rows)*100)
    ##print("total nr of lost msgs: ",loss)
    #print("percentage of lost msgs: ", loss/(len(rows)+loss)*100)
    #print("longest loss: ", longest_loss)
    return loss/(len(rows)+loss)*100
    



#FILE = glob.glob("DATA/*.hdf5")
#for d in FILE:
#    f = h5py.File(d, 'r')
#    loss = []
#    for ID in list(f.keys()):  # iterate over devices
#        if ID == "LOGs" or ID == "Flags":
#            continue
#        print("f", f, "ID", ID)
#        a = np.array(f[ID]['data'][:]) 
#        ordered = order(a.tolist())
#        fs = f[ID].attrs["sampling rate"]
#        print("fs", fs)
#        getError(ordered, fs)
#        print("####")
