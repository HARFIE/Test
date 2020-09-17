import json
import os, os.path
import numpy as np
import pandas as pd
import time as t
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from torch.utils.data import Dataset
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, Input
from tensorflow.keras.models import model_from_json

import tensorflow.nn as tf_nn
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCC
#from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy as STKCA
from tensorflow.keras.metrics import SparseCategoricalCrossentropy as SCCA
from tensorflow.keras.models import load_model

import pickle
from tensorflow.keras import Model
import networkx as nx
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import KFold, StratifiedKFold
import csv
from tensorflow import keras as keras

tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#keras.losses.SparseCategoricalCrossentropy()

from matplotlib import pyplot as plt


from threading import Event

MYJSONS = 'C:\\Users\\up201\\Desktop\\Tese\\openposeOriginal\\openpose-master\\examples\\media\\myjson'

#prints the values of pose_keypoints_2d
def printList(a):
    #print('starting...')
    for x in range(len(a)//3):
            #print("%s %s"% (a[x], a[x+1]))
            print("%s %s %s" % (a[x*3], a[x*3+1], a[x*3+2]))

def buildVal(data):
    x=0
    val= np.array([[data[x*3], data[x*3+1], data[x*3+2]]])
    for x in range(1,len(data)//3):
        val2 = np.array([[data[x*3], data[x*3+1], data[x*3+2]]])
        val = np.vstack((val, val2))
    return val

#open json and and feed pose_keypoints_2d to printList
def openJson(df, Jpath, vname, jname):
    Jpath_name = Jpath + vname+'_rgb_'+ jname + '_keypoints.json'
    with open(Jpath_name) as f:
        data = json.load(f)
    
    pep = data['people']
    #printList(pep[0]["pose_keypoints_2d"])

    #build np araay of values with 3 columns
    val = buildVal( pep[0]["pose_keypoints_2d"])    
    
    df2 = pd.DataFrame([[vname[0:4], vname[4:8], vname[8:12], vname[12:16], vname[16:20], jname, val]], columns = ['Subject', 'Camera','Position', 'R', 'Activity', 'json name' ,'data'])
    df=df.append(df2, ignore_index=True)

    return df
def openJson2(df, Jpath, jname):

    Jpath_name = Jpath + jname + '_keypoints.json'
    with open(Jpath_name) as f:
        data = json.load(f)
    
    pep = data['people']
    #printList(pep[0]["pose_keypoints_2d"])
    
    #build np araay of values with 3 columns
    if(len(pep)>0):
        val = buildVal( pep[0]["pose_keypoints_2d"])    
        df2 = pd.DataFrame([[ val, jname]])
        #print(df2[1].tolist())
        #print('df2 ..... iloc[:,0]')
        #print(df2[0].tolist())
        df2 = expand_RL(df2[0], df2[1])
        #df=df.append(df2, ignore_index=True)
        df= df2
    return df
    
    #df=df.append()
def parseVName(vname):
    s= vname[0:4]
    print(s)
    

#open json folder and feed every json in folder to openJson 
def readJsonInFolder(df, DIR_JF_path, folder):
    listD = os.listdir(DIR_JF_path) # dir is your directory path
    number_files = len(listD)
    #print (number_files)  
    for x in  range(number_files):
        
        st= folder +'_rgb_' + f'{x:012}' +'_keypoints'
        #print (st)
        df= openJson(df, DIR_JF_path+'\\', folder, f'{x:012}' )
    return df
    
#limites:
#A - 1 --
#C - 3
#P - 8
#R - 2
#A - 60
#builds folder name per video
def buildDataSetFName(n):
    a = (n-1) % 60 + 1
    r = ((n-1) // 60 ) % 2 +1
    p = ((n-1) //(60*2) ) % 8 +1
    c = ((n-1) // (60*2*8) ) % 3 +1
    #if(a==0):
    #    a=60
    #if(r==0):
    #    r=2
    #if(p==0):
    #    p=8
    #if(c==0):
    #    c=3
    A= f'{a:03}'
    R= f'{r:03}'
    P= f'{p:03}'
    C= f'{c:03}'
    
    finalString= 'S001C' + C + 'P' + P + 'R' + R + 'A' + A
    #print(  "N=%s C=%s P=%s R=%s A=%s" %( n, C, P, R, A))
    return (finalString)
def printEvJs_in_Fold(df, json_Dir,N, startT):
    for n in range(1, N+1, 1):
        folder = buildDataSetFName(n)
        path2FW = json_Dir + '\\' + folder +'_rgb'
        df = readJsonInFolder(df, path2FW, folder)
        
        if(n%10 == 0):
            ETA = getETA((t.time_ns()-startT)/1000000, getPercentage(n,N))
            print('%d \t %.2f%% \t ETA= %s ms ---- ETA2= %s ms' % (n,getPercentage(n,N), ETA, buildTime(2078*(N-n))))
        #print(folder)
    return df


#time functions
def getPercentage(n, T):
	return 100*n/T
def getETA(elapTime, percent):
	ms = elapTime*100/percent-elapTime
	return buildTime(ms)
def buildTime(ms):
    Hour= getHour(ms)
    Min = getMin(ms)
    Sec = getSec(ms)
    Ms = getMs(ms)
    return f'{int(Hour)}:{int(Min)}:{int(Sec)} - {int(Ms):03}'
def getSecFms(ms):
    return ms/1000
def getHour(ms):
    Vhour=ms/(1000*60*60)
    return Vhour
def getMin(ms):
    Vmin= getHour(ms)-math.floor(getHour(ms))
    return Vmin * 60
def getSec (ms):
    Vsec = getMin(ms)- math.floor(getMin(ms))
    return Vsec * 60
def getMs(ms):
    Vms= getSec(ms)- math.floor(getSec(ms))
    return Vms*1000
def getAvg(totalDur_ms, nElem ):
    fAvg = totalDur_ms/nElem
    return fAvg
def ns2ms(ns):
    return ns/1000000

#formatting
def form2dec(number):
    return  f'{number:.2f}'

#panda manipulation
def joinLabels(Ldf, AJdf):
    a= AJdf.sort_values(by=['Activity', 'Camera' , 'Position', 'R', 'json name'])
    b= Ldf

    for i in range(0, len(b.index)):
        a.loc[a['Activity'] == b.iloc[i,0], 'Label']=b.iloc[i,1]
    #a = a.drop(columns =['data', 'json name'])
    #d = a.drop_duplicates(subset = 'Activity', keep = 'first', inplace = True) 
    
    return a

def getVideoJsons(data, n):
	A, R ,P, C = getVideoName(n)
	headers = list(data.columns)
	video1 = data.loc[(data[headers[4]]== 'A'+formatVideoName(A)) & 
		(data[headers[3]]     == 'R'+formatVideoName(R)) &
		(data[headers[2]] == 'P'+formatVideoName(P)) &
		(data[headers[1]]  == 'C'+formatVideoName(C))]
	return video1
#Feature Extraction
def build_feat1(data, n):
	
	#print(data)
	
	video1 = getVideoJsons(data, n)

	feature=pd.DataFrame()
	feat2 = pd.DataFrame(feature, columns = ['Shift'])
	video1_data = video1['data']
	
	for i in range(1, len(video1_data.index-1) ):
		now= np.asarray(video1_data.iloc[i])
		before = np.asarray( video1_data.iloc[i-1])

		val = np.subtract(now[:,0:2]*np.transpose(now[:,2][np.newaxis]), before[:,0:2]*np.transpose(before[:,2][np.newaxis]))

		#print(i)
		feature= feature.append(pd.DataFrame([pd.DataFrame(val, columns = ['Shift_X', 'Shift_Y'])], columns = ['Video'], index =  ['Video'+str(n)]), ignore_index = False)
	return feature

def getVideoName(n):
    #print(n)
    a = (n) % 60 + 1
    #print(a)
    r = ((n) // 60 ) % 2 +1
    p = ((n) //(60*2) ) % 8 +1
    c = ((n) // (60*2*8) ) % 3 +1

    return a,r,p,c

def getVideoNumber(c, p, r, a):
	N= (c-1)*(8 * 2 * 60) +(p-1)*(2*60) + (r-1)* 60 +a 
	return N
def formatVideoName(a):
    A= f'{a:03}'
    #R= f'{r:03}'
    #P= f'{p:03}'
    #C= f'{c:03}'
    return A

def getAction(n):
	a = (n-1) % 60 + 1
	return a

def getActions(alist):
	#if (alist[0]==0):
		#alist = [a+1 for a in alist ]
	#	pass
	print(alist)
	for a in range(0,len(alist), 1 ):
		alist[a] = getAction(alist[a])
	return alist


def getSameActionVideos(n):
	Li=[]
	N=getAction(n)

	for i in range(0, 2880//60, 1):
		Li.append(N+i*60)
	return Li 

def feat1NVideos(path, N):
	test = pd.read_json(path, orient='split')
	headers = list(test.columns)
	data = test[headers[0:5]+headers[6:7]]
	feature = pd.DataFrame()
	startT= t.time_ns()
	for i in range(1,N+1,1):
		feature =  build_feat1(data,i)
		feature.to_json(MYJSONS + '\\' + 'feature1' + '\\'+ 'Video' + f'{i:04}' + '.json', orient = 'split')
		if (i%10 == 0):
			ETA = getETA((t.time_ns()-startT)/1000000, getPercentage(i,N))
			print('%d \t %.2f%% \t ETA= %s ms ---- ETA2= %s ms - - - - - - - elapsed_Time= %s' % (i,getPercentage(i,N), ETA, buildTime(87*(N-i)), buildTime(ns2ms(t.time_ns()-startT))))
	return feature , test


def loadFeature1(n):
	path = MYJSONS + '\\' + 'feature1'+ '\\'+ 'Video'+ f'{n:04}' + '.json'
	F = pd.read_json(path, orient = 'split')
	return F

#training

def labelsPerIndex(mat , label):
	N= np.size(mat, 0)
	#print(list(mat))
	List=[]
	for i in range(0,N):
		List.append(label[getLabelIndex(mat[i]+1)])

	return List

def getLabelIndex(n):	
	return (n-1)%60

def jsonNameIndex(jNlist):
	count = -1
	a = []
	for n in range(len(jNlist)): #len(jNlist)
		if(jNlist[n]== 0):
			count = count+1
		a.append(count)
	return a

def build_test(Total,testSize):
	space = math.floor(Total/testSize)
	#print(space)
	count = 0
	test = [count]
	train = list(range(0, Total,1))

#pop

	for i in range(0,testSize-1):
		train.remove(test[i])
		test.append(count + space)
		count = count +space
	return train, test

def gatherVideoJsons(test, test_ind):
	LL = []
	# print('My func : ')
	for x in test_ind:
		vno = x
		# print(vno)
		LL.extend(test.index[test[-1] == vno].to_list())
	return LL

def openData(data):
	p1 = data[0]
	p2 = data[1]
	return p1,p2

def expandDF(element, label):
	#print('----------- EXPAND DF --------------')
	#print(element)
	#print(len(element))
	b= np.asarray(element)
	c = b[:, 0:2].reshape(50)
	c=c.tolist()
	c.append(label)
	return c
def expand (X, y):
	print('----------- EXPAND --------------')
	print(X)
	print(y)
	a= X.iloc[0,0]
	c= expandDF(a, y[0])
	t1 = pd.DataFrame([c])
	final = len(X.index)
	for i in range(1,final):
		a= X.iloc[i,0]
		b = y.iloc[i]
		c = expandDF(a,b )
		t1 = t1.append([c], ignore_index = False)
		if (i % 100 == 0):
			print(f'{i}' + ' / ' + f'{final}' + '\t' + f'{getPercentage(i, final): .02}' + ' %')
	t1.index = X.index

	return t1

def expand_RL(elements, label):
	a=elements[0]
	v=expandDF(a,label)
	df = pd.DataFrame([v])
	#print(df)
	return(df)


class NP_2DBlock:
    def __init__(self, rows, cols, fil_Name, name, gen):
        self.data= np.zeros((rows,cols))
        self.fileName = fil_Name
        self.fileTXT = 'TXT'+self.fileName[3:-3]+'txt'
        self.name = name
        self.headers = []
        self.generator = gen
        self.shape = (rows,cols)
    def __str__(self):

        data = f'{self.data[:10,:]}'
        rep = '\n\t'+self.name+'-->'+ self.fileName + '\n'+'Type: '+ f'{type(self.data)}'+ '\tShape: '+ f'{self.shape}'  +'\n' + f'{self.headers}'+'\n' + data
        return rep

    def __getitem__(self, coords):
        return self.data[coords]

    def generate(self, *args):
        self.data = self.generator(*args)
        self.shape = self.data.shape

    def setCol_names(self, names):
        self.headers = names

    def loadData(self, fil_Name):
        self.data = np.load(fil_Name)
        self.shape = self.data.shape

    def load(self, fil_Name):
        with open(fil_Name, 'rb') as fp:   # Unpickling
            DB = pickle.load(fp)
        return DB
    def save(self, fil_Name = None):
        if not fil_Name:
            fil_Name = self.fileTXT
        print(fil_Name)
        with open(fil_Name, 'wb') as fp:   #Pickling
            pickle.dump(self, fp)

class mDataset(Dataset):

    def __init__(self, jName):

        test= pd.read_json(jName,orient= 'split')
        xyz = test.to_numpy()
        self.n_samples = xyz.shape[0]

        # note that we do not convert to tensor here
        fl = list(range(50,100))+ list(range(150,200))
        self.features = xyz[:,fl]
        self.labels = xyz[:, [202]]
        self.jno = xyz[:, [200]]
        self.vno = xyz[:, [201]]        
        self.labEnc=np.array(test[202].unique())
        self.labelsi = np.zeros(shape = self.labels.shape[0])
        for i in range(len(self.labels)):
        	self.labelsi[i] = np.where(self.labEnc == self.labels[i])[0]

        #self.transform = transform

    def __getitem__(self, index):
        sample = self.features[index], self.jno[index], self.labelsi[index]
        #print('features: ',sample[0].shape)
        #if self.transform:
        #    sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class pDataset(Dataset):
    def __init__(self,  fil_Name=None, description = None):
        if fil_Name:
            xyz = np.load(fil_Name)
        else:
        	print('File Name not provided!')
        	return -1
        #DBtest= mL.pd.read_json(ndsname, orient= 'split')
        #xyz = DB.data
        self.n_samples = xyz.shape[0]


        self.features= xyz[:,:-2]
        self.jno = xyz[:, -2]
        self.y_data = xyz[:,-1]

        if description == 'POS':
            self.features = self.features[:,:30]
        elif description == 'JOD':
            self.features = self.features[:,30 : 30 + 630 ] # + 364 + 756
        elif description == 'JLD':
            self.features = self.features[:,30 + 630  : 30 + 630 + 364 ] # + 364 + 756
        elif description == 'LLA':
            self.features = self.features[:,30 + 630 +364 : 30 + 630 + 364+ 756 ] # + 364 + 756
        pno = np.zeros((self.n_samples,1))
        ind1 = np.where(self.y_data>=49)
        pno[ind1] = 1
        self.features = np.hstack((self.features, pno))

        self.beg = np.where(self.jno == 0)[0]
        # note that we do not convert to tensor here
        #self.Pos_data = xyz[:, :50   ]
        #self.Mov_data  = xyz[:, 100:150]
        #self.vno      = xyz[:, [200]  ]
        #self.jno 	  = xyz[:, [201]  ]
        #self.y_data   = xyz[:, [202]  ]
        #self.n_pep	  = xyz[:, [203]  ]

        #self.features = np.hstack((self.Pos_data, self.Mov_data))#, self.n_pep ) )

    def __getitem__(self, index):
        
        sample = self.features[index], self.y_data[index]

        return sample

    def __len__(self):
        return self.n_samples

def build_graph():
	bodyParts = range(0,25,1)
	bones = [(0,15), (0,16), (0,1),
		 (1, 2), (1,5), (1,8),
		 (2,3), (3,4),
		 (5,6), (6,7),
		 (8,9), (8,12),
		 (9,10), (10,11),
		 (11,24), (11,22), (22,23),
		 (12,13), (13,14),
		 (14, 21), (14, 19),
		 (19,20),
		 (15,17), (16,18)]

	G=nx.Graph()
	G.add_nodes_from(bodyParts)
	G.add_edges_from(bones)
	return G

def buildTestSet(nvideos, nversions, begining, lengths, nsteps):
    test_index = np.zeros((nvideos,))
    All = np.sum(lengths).astype(int)-nvideos*nversions*(nsteps-1)

    all_arr = np.array(range(All))
    for video in range(nvideos):
        version = video % nversions
        index =  video + version * nvideos
        test_index[video] = index
        
    test_index = test_index.astype(int)
    lengths_sum =0

    for n in test_index:
        lengths_sum += lengths[n]

    lengths_sum = lengths_sum.astype(int)
    test_set = np.zeros((lengths_sum-nvideos*(nsteps-1),))
    ptr = 0
    lengths = lengths.astype(int)


    for n in test_index:
        rang = np.array(range(begining[n]-n*(nsteps-1), begining[n] - n*(nsteps-1) + lengths[n] - (nsteps-1)))
        test_set[ptr : ptr + lengths[n] - (nsteps-1)]= rang
        ptr += lengths[n] - (nsteps-1)
     
    #print(test_index.shape, ptr)
    train_set = np.setdiff1d(all_arr, test_set)
    return test_set.astype(int), train_set.astype(int)

def forExtract(cl, iteri, ns, l):
    ret = [None]* len(iteri)
    inis = np.zeros((len(iteri)))
    fins =  np.zeros((len(iteri)))
    for n,i in enumerate(iteri):
        ini = cl[i].astype(int) - i*(ns-1)
        fin = ini + l[i].astype(int) - (ns)+1
        ret[n] = range(ini, fin)
        inis[n] = ini
        fins[n]= fin
    #print(len(ret), ret[0])
    #print('\t\tmaxs\t',np.max(ini), np.max(fin))
    mi = np.max(ini)
    mf = np.max(fin)
    #print(ret)
    #input()
    return np.array(ret), mi, mf


def expandSplit(compact):
    comp_LenSum = np.sum(np.array([len(k) for k in compact]))
    exp_set = np.zeros((comp_LenSum))
    counter = 0

    for k in range(compact.shape[0]):
        rang = range(counter, counter + len(compact[k]))
        #print(f'Test[{k}]:', len(rang), len(test[k]))
        exp_set[rang] = list(compact[k])
        counter += len(compact[k])
    return exp_set.astype(int)




def extractSplit(nsteps, lengths, e ):
    cum_length = np.zeros(lengths.shape[0]+1)
    for i,l in enumerate(lengths):
        cum_length[i+1] += l + cum_length[i] 

    eval_e, miev, mfev = forExtract(cum_length, e, nsteps, lengths)

    eval_set = expandSplit(eval_e)

    return eval_set.astype(int)-1

def extractSplits(nsteps, lengths, n, m ):

    #print(type(n), type(m), type(n[0]), type(m[0]))
    mi=m[0]
    ni=n[0]
    #print(mi*lengths[mi])
    #print(type(lengths))
    #print(type(lengths[mi]))

    #print(f'Test: {  mi*lengths[mi].astype(int) - mi*(nsteps-1) }  , { mi*lengths[mi].astype(int) - mi*(nsteps-1) + lengths[mi].astype(int) - (nsteps-1) }' )
    #print(f'Train: { ni*lengths[ni].astype(int) - ni*(nsteps-1) }  , { ni*lengths[ni].astype(int) - ni*(nsteps-1) + lengths[ni].astype(int) - (nsteps-1) }' )
    cum_length = np.zeros(lengths.shape[0]+1)
    for i,l in enumerate(lengths):
        cum_length[i+1] += l + cum_length[i] 

    #print(f'lens in: te {len(m)}, tr{len(n)}')

    test  , mite, mfte= forExtract(cum_length, m, nsteps, lengths)
    train , mitr, mftr= forExtract(cum_length, n, nsteps, lengths)
    #print(f'\t\t\t\tMAX\t {np.max([mite, mitr])}, {np.max([mfte, mftr])}')
    durs= 0


    # test = np.array([range(mi*lengths[mi].astype(int) - mi*(nsteps-1), mi*lengths[mi].astype(int) - mi*(nsteps-1) + lengths[mi].astype(int) - (nsteps-1)) for mi in m])
    # train = np.array([ range( ni*lengths[ni].astype(int) - ni*(nsteps-1), ni*lengths[ni].astype(int) - ni*(nsteps-1) + lengths[ni].astype(int) - (nsteps-1) ) for ni in n ])

    test_LenSum = np.sum(np.array([len(k) for k in test]))
    print('test_LenSum: ',test_LenSum, len(test), len(test[0]))
    test_set = np.zeros((test_LenSum))
    counter = 0

    for k in range(test.shape[0]):
        rang = range(counter, counter + len(test[k]))
        #print(f'Test[{k}]:', len(rang), len(test[k]))
        test_set[rang] = list(test[k])
        counter += len(test[k])

    train_LenSum = np.sum(np.array([len(k) for k in train]))
    #print(train_LenSum, len(train), len(train[0]))
    train_set = np.zeros((train_LenSum))
    counter = 0
    for k in range(train.shape[0]):
        rang = range(counter, counter + len(train[k]))
        #print(f'Train[{k}]:', len(rang), len(train[k]), np.max(train[k]))
        train_set[rang] = list(train[k])
        counter += len(train[k])
    #print(train_set.shape, test_set.shape)
    #print(f'lens: {train_set.shape} {test_set.shape}')
    #print(f'max: {np.max(train_set)} {np.max(test_set)}\n')
    
    return train_set.astype(int)    , test_set.astype(int)
    

def getRandom(na, nv):
    random = (np.random.rand((na))*nv).astype(int)
    return random


def getTuple(size, shapes, tfilter, tfilter_i):
    dim_list = [None]*size
    for i,n in enumerate(shapes):
        dim_list[i] = list(range(n))
    dim_list[tfilter_i] = tfilter
    return tuple(dim_list)


def getSplitSetsBasic4(shapes, testfilter_i, testfilter):

    All_len = 1
    for n in shapes:
        All_len *= n 
    All = np.arange(All_len).reshape(shapes)

    max_i = shapes[testfilter_i]
    All_i = set(range(max_i))
    train_filter = All_i - set(testfilter)


    test = All[:,testfilter,:,:].flatten('C').astype(int)

    train = All[:,list(train_filter),:,:].flatten('C').astype(int)

    return [(train, test)]

def get_testInd(n_a, n_v, n_s, lab, withFold= False): #360 * 8

    # 3C 8P 2R 60A
    sh= np.array(lab).shape
    myLab = np.arange(sh[0]*sh[1])
    myLab = myLab.reshape((3,8,2,60))
    #print(myLab[:,:,:,6].flatten)
    

    test_ind = np.zeros((n_v//n_s - 1, n_a * n_s))
    eval_ind = np.zeros((1,n_a*n_s))

    for ns in range(n_v//n_s):
        test = [None] * n_a * n_s
            
        for s in range(n_s):
            ver = len(lab[0])    #
            random = getRandom(n_a, ver)


            for n,rand in enumerate(random):
                test[n+n_a*s] = lab[n].pop(rand)

        if ns == 0:
            eval_ind[ns,:] = np.array(test).astype(int)
        else:
            test_ind[ns-1,:] = np.array(test).astype(int)

        
    return test_ind, eval_ind

def buildInputList(n_a, n_v):
    labels = [None]*n_a

    for n in range(n_a):
        labels[n]=[None]*n_v

    n=0
    for vers in range(1, n_v+1):
        for act in range(1, n_a+1):
            labels[ act-1][ vers-1]= n
            n+=1
    #print('labels:', labels)        
    return  np.array(labels).astype(int)


def split(test, All, eval_i, withFold= False):
    #print(f'test: {test.shape}')  
    #print(f'All: {len(All)}')
    nfold = test.shape[0]
    ret = [None]* nfold
    
    eval_set = set(eval_i[0,:])
    
    for f in range(nfold):
        test_set = set(test[f,:])
    
        train_set = All- test_set -eval_set
    
        ret[f]=  (train_set, test_set)

    return ret, eval_set

def buildLengths(downsample):

    PNR_fname = 'TXT/PersonNReg.txt'
    with open(PNR_fname, 'rb') as fp:   # Unpickling
                personNReg = pickle.load(fp)
    
    lengths = np.zeros((len(personNReg),))
    for n in range(len(personNReg)):
        b = len(personNReg[n]) % downsample !=0
        lengths[n] = len(personNReg[n])//downsample + b
    return lengths