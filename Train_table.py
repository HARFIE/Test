import myLib as mL
from myLib import NP_2DBlock 
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import psutil
import csv
import vizualizeAccLoss as vAL
import test as t
import PLSTM as PA
mL.tf.keras.backend.clear_session()

#jname= 'input_201.json'
#arraysaved = 'lstm_input30.npy'
#tgsaved = 'lstm_target_t11.npy'
DB_name = 'NPY/DataBlock5.npy'


#print(DB_name[4:-4])
mL.tf.get_logger().setLevel('INFO')
mL.os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def getvnoFsno(sno, pNR):
    jcounter =0
    for n in range(len(pNR)):
        if sno < jcounter + len(pNR[n]):
            #print(f'{jcounter} --- {sno} --- {jcounter + len(pNR[n])}')
            #A,_,_,_ = mL.getVideoName(n)
            return n
        jcounter += len(pNR[n])

def getfeat(datast, n,s, ds):

    feat,_= datast[n+1-s*ds:n+1:ds] 
    #print('feat', feat.shape)
    aux = len(datast)
    if feat.shape[0]!= 11:
        print('GET FEAT: ', n+1-s*ds, n+1, ds, '----', feat.shape, aux)
    return feat

    #return (mL.np.random.rand(1,datast)*60)//10+1

def buildInput(samples, steps, features, targ, dataset, pnr, ds, ds_m, verbose = 0):

    print('Creating m_input')

    startb = mL.t.time_ns()
    tdur=0
    jcount = 0
    icount=0
    ilen = 0

    pcount = 0
    for n in range(len(pnr)):   #2880
        action ,_,_,_ = mL.getVideoName(n)

        pcount += len(pnr[n])
        for j in range(0,len(pnr[n]), ds):
            if j>= steps*ds-1:
                ilen +=1
    print(ilen)
        
            

    nf = mL.np.zeros((ilen,steps,features))
    tg = mL.np.zeros((ilen))

    jcount=0 
    
    for n in range(len(pnr)):   #2880
        action  ,_,_,_= mL.getVideoName(n)
        
        for j in range(0,len(pnr[n]),ds):
            if j>= steps*ds-1:
                feat1 = getfeat(dataset, jcount, steps, ds)
                if verbose:
                    print('....... Building sample in time window ..........')
                    print('feat1',type(feat1), feat1.shape, '\n','nf', type(nf), nf.shape, '\n' )        
                
                nf[icount] = feat1
                _, tg[icount] = dataset[jcount]
                #print('target',tg[icount], icount, jcount)
                icount +=1
            
                if icount % 10000 ==0 :
                    dur1 = mL.t.time_ns() -startb
                    tdur += dur1
                    #print(jcount, j, n)

                    print('Building Input ....',icount+1,'/', ilen,'\tElapsed Time', mL.buildTime(mL.ns2ms(tdur)),'ms', 
                            '\tAverage:', mL.ns2ms(tdur//(icount+1)),'ms', '\tTime interval', mL.buildTime(mL.ns2ms(dur1)),'ms' ) 
                    
                    print('------------------------------------------------------------------\n\n')
                    startb = mL.t.time_ns()


                if verbose:
                    print('..........  Adding  sample to array  ..............')
                    print('nf\n', nf)

            jcount +=  ds
        #jcount += len(pnr[n])
        
    if verbose:
        print( '\n\n\t\tFinished Building Input\n')

        feat1 = nf[0,0]
        tf = nf[0]
        print('feat1',type(feat1), feat1.shape, '\n', feat1)
        print('\ntf',type(tf), tf.shape, '\n',tf)
        print('\nnf',type(nf), nf.shape, '\n',nf)

    #print( mL.np.isnan(nf).any() )
    #print( mL.np.isnan(tg).any() )
    print(f'nf samples: {nf.shape[0]}, icount: {icount}')
    return nf, tg

def buildModel(nf, ns, nt, mname):

    modl = mL.Sequential()  
    #print(nt)
    # modl.add(mL.tf.compat.v1.keras.layers.CuDNNLSTM(100, input_shape=(ns, nf), return_sequences=True))
    # modl.add(mL.tf.compat.v1.keras.layers.CuDNNLSTM(100, input_shape=(ns, ns)))
    # modl.add(mL.keras.layers.Dropout(0.5))
    # modl.add(mL.keras.layers.Dense(100, activation=mL.tf.nn.relu))
    # modl.add(mL.keras.layers.Dense(nt, activation=mL.tf.keras.activations.softmax)) # mL.tf_nn.sigmoid

 
    #previous : 
    lstm_out = 30
    #modl.add(mL.Dense(nf, input_shape=(ns,nf),activation=mL.tf_nn.softmax )) # mL.tf_nn.sigmoid
    
    modl.add(mL.tf.compat.v1.keras.layers.CuDNNLSTM( lstm_out, input_shape=(ns, nf), return_sequences=True))
    #modl.add(mL.LSTM( 256, input_shape=(ns, nf), return_sequences=True))
    #modl.add(mL.Dropout(0.5))
    #modl.add(mL.BatchNormalization())

    #modl.add(mL.tf.compat.v1.keras.layers.CuDNNLSTM( lstm_out, input_shape=(ns, nf), return_sequences=True))
    #modl.add(mL.LSTM( 256, input_shape=(ns, nf), return_sequences=True))
    #modl.add(mL.Dropout(0.5))
    #modl.add(mL.BatchNormalization())

    modl.add(mL.tf.compat.v1.keras.layers.CuDNNLSTM( lstm_out  , input_shape=(ns, nf)        ))    
    #modl.add(mL.LSTM( 128  , input_shape=(ns, nf)        ))    

    modl.add(mL.Dropout(0.5))
    #modl.add(mL.BatchNormalization())

    modl.add(mL.Dense(lstm_out, activation=mL.tf_nn.relu))
    modl.add(mL.Dense(nt, activation=mL.tf.keras.activations.softmax )) # mL.tf_nn.sigmoid


    #model.add(mL.LSTM((100), input_shape=(2, 50),return_sequences=True))
    #model.add(mL.Dense((50)))


    lfn= mL.SCC()
    m = mL.STKCA(k=1)
    modl._name = mname
    modl.compile(loss=lfn, optimizer='adam',metrics=[m])
    return modl

def printAcc(hist):
    print('Printing Accuracy graph ...\n')

    mL.plt.plot(hist.history['sparse_top_k_categorical_accuracy'])
    mL.plt.plot(hist.history['val_sparse_top_k_categorical_accuracy'])
    mL.plt.title('model accuracy')
    mL.plt.ylabel('accuracy')
    mL.plt.xlabel('epoch')
    mL.plt.legend(['train', 'test'], loc='upper left')
    mL.plt.show()

def printLoss(hist):
    print('Printing Loss graph ...\n')

    mL.plt.plot(hist.history['loss'])
    mL.plt.plot(hist.history['val_loss'])
    mL.plt.title('model Loss')
    mL.plt.ylabel('Loss')
    mL.plt.xlabel('epoch')
    mL.plt.legend(['train', 'test'], loc='upper left')
    mL.plt.show()

def printConfusionMatrix(cm_targ, cm_pred):
    ab=confusion_matrix(cm_targ, cm_pred)
    print(f'shape ab:{ab.shape}\t targ shape:{cm_targ.shape}\t pred shape:{cm_pred.shape}')

    print(len(set(list(cm_targ))))
    #au =  mL.np.array(list(set(list(cm_targ)))).astype(int)
    au =( mL.np.arange(ab.shape[0]), mL.np.arange(ab.shape[1]))
    print('au', au[0].shape, au[1].shape)
    df_cm = mL.pd.DataFrame(ab, columns=au[0], index = au[1])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    mL.plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap= mL.plt.get_cmap('RdYlGn'))
    mL.plt.show()

def printHeatMap(hm_pred):
    ax= sns.heatmap(hm_pred, cmap= mL.plt.get_cmap('RdYlGn'))
    mL.plt.show()

def printOutLabels(ol_labels,i, t):
    rr = mL.pd.DataFrame()

    li_lab = [n for n in ol_labels.iloc[:,1]]
    predi = [li_lab[n] for n in i]
    expec = [li_lab[n] for n in t.astype(int)]

    rr['Predict'] = predi
    rr['Expected']= expec

    #               Prety Print for Predicted vs Expected
    print('li_lab', type(li_lab), len(li_lab))
    print(rr)

def printScatter(sc_targ, sc_pred):
    mL.plt.scatter(range(sc_targ.shape[0]), sc_targ, c='g')
    mL.plt.scatter(range(sc_targ.shape[0]), sc_pred, c='r')
    mL.plt.show()

def buildModelName( fe,description=None):
    if description != None:
        return f'model_{description}_f{fe}'
    else:
        return f'model_f{fe}'

def buildInputName( nf, nt, ds, description=None):
    if description != None:
        return f'input_{description}_F{nf}_T{nt}_D{ds}.npy'
    else:
        return f'input_{description}_F{nf}_T{nt}_D{ds}.npy'

def buildTargetName( nf, nt, ds,description=None):
    if description != None:
        return f'target_{description}_F{nf}_T{nt}_D{ds}.npy'
    else:
        return f'target_F{nf}_T{nt}_D{ds}.npy'

def createDir(dir_path):
    try:
        mL.os.mkdir(dir_path)
    except OSError:
        print (f'Creation of the directory {dir_path} failed.')
    else:
        print (f'Successfully created the directory {dir_path}.')

def buildModelWorkspace( fe, fname,description = None):
    model_Folder = 'models\\training_model\\'
    parent_path  =  model_Folder + fname
    input_path   =  parent_path  + '\\Input'
    target_path  =  parent_path  + '\\Target'
    check_path   =  parent_path  + '\\Checkpoints'
    fold_path    =  parent_path  + '\\Folds'
    fold_in_path = input_path + '\\Folds'
    
    if mL.os.path.isdir(parent_path):
        
        if not(mL.os.path.isdir(input_path)):
            createDir(input_path)
            createDir(fold_in_path)
        else:
            if not(mL.os.path.isdir(fold_in_path)):
             createDir(fold_in_path)   
        
        if not(mL.os.path.isdir(target_path)):
            createDir(target_path)
        
        if not(mL.os.path.isdir(check_path)):
            createDir(check_path)
        
        if not(mL.os.path.isdir(fold_path)):
            createDir(fold_path)
        

    else:
        createDir(parent_path)
        createDir(input_path)
        createDir(target_path)
        createDir(check_path)
        createDir(fold_path)
        createDir(fold_in_path)

    return input_path, target_path, check_path, parent_path, fold_path

def onehotencode(data):
    zeros = mL.np.zeros(data.shape)
    i =mL.np.argmax(data ,axis = 1)
    for r in range(data.shape[0]):
        zeros[r,i[r]] =1
    return zeros
#teste_numpy 

class My_Input_Generator(mL.tf.keras.utils.Sequence) :
  
  def __init__(self, index, inp, targ, batch_size) :
    self.index = index
    self.input = inp
    self.target = targ
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (mL.np.ceil(len(self.index) / float(self.batch_size))).astype(mL.np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.input[self.index[idx * self.batch_size : (idx+1) * self.batch_size], :, :]
    batch_y = self.target[self.index[idx * self.batch_size : (idx+1) * self.batch_size]]
    
    #print('\n', batch_y[:1000:100])
    return mL.np.array(batch_x), mL.np.array(batch_y)

class My_Validation_Generator(mL.tf.keras.utils.Sequence) :
  
  def __init__(self, index, inp, batch_size) :
    self.index = index
    self.input = inp
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (mL.np.ceil(len(self.index) / float(self.batch_size))).astype(mL.np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.input[self.index[idx * self.batch_size : (idx+1) * self.batch_size], :, :]
    
    #print('\n', batch_y[:1000:100])
    return (mL.np.array(batch_x),)

def train_model(train_set, test_set, m_input, target, foldno, fold_path, par_path, n_epoch, batch_size, mname, process, model, ds, withFold = False):
    last_tr_end, last_te_end = 0 , 0
    print('train_set', len(train_set))
    print('test_set' , len(test_set))
    
    train_generator = My_Input_Generator(train_set, m_input, target, batch_size)
    #train_ds = mL.tf.data.Dataset.from_generator(lambda: My_Input_Generator(train_set, m_input, target, batch_size),(mL.tf.float32,mL.tf.float32), output_shapes = ((None,m_input.shape[1],m_input.shape[2]),(None,)))
    #train_input = m_input[train_set[last_tr_end:]  ,:,:]
    #train_target =  target[train_set[last_tr_end:]]
    print('train dataset')
    
    test_generator =  My_Input_Generator(test_set, m_input, target, batch_size)
    #test_ds = mL.tf.data.Dataset.from_generator(lambda: My_Input_Generator (test_set, m_input, target, batch_size), (mL.tf.float32,mL.tf.float32), output_shapes = ((None,m_input.shape[1],m_input.shape[2]),(None,)) )
    #test_input = m_input[test_set[last_te_end:],:,:]
    #test_target =  target[test_set[last_te_end:]]
    print('test dataset')


    
    #print(train_ds[0])


    print('extracted')
    checkpoint_filepath= 'Model_{epoch:02d}-{sparse_top_k_categorical_accuracy:.3f}' # sparse_top_k_categorical_accuracy
    
    checkpoint = mL.ModelCheckpoint(fold_path + '\\final_{}.h5'.format(checkpoint_filepath, monitor= 'val_sparse_top_k_categorical_accuracy', verbose = 1, save_best_only=True, mode = 'max')) #val_sparse_top_k_categorical_accuracy
    csv_Logger = mL.CSVLogger(fold_path + f'\\final_history.csv', append = True, separator = ',')

    #input()
    start = mL.t.time_ns()
    h=model.fit(  x=train_generator, 
        epochs=n_epoch, verbose=1,
        validation_data = test_generator,
        callbacks = [checkpoint, csv_Logger] )
    dur = mL.t.time_ns()-start
    print('Finished Training --- ',mL.buildTime(mL.ns2ms(dur)),'ms')

    model.save(mname+'.h5')
    print("Saved model to disk")

    memory = mL.np.array([process.memory_info().rss/(1024*1024)])
    print('Memory occupied:  ' + f'{memory[0]:.2f}' + ' MB')
    
    print('par_path: ', par_path)
    print('fold_path: ', fold_path)
    with open(fold_path + '\\Memory.csv', 'a+') as csvfile:
        mL.np.savetxt(csvfile, memory)
    
    return model

def saveNPY(folder, data, verbose=False):
    fold_start = mL.t.time_ns()
    mL.np.save(folder , data )
    fold_dur = mL.t.time_ns() - fold_start
    print(f'{folder}\t Saving duration: {mL.buildTime(mL.ns2ms(fold_dur))} ms')

def  getloadmodelName(path, sf, ds):
    pref = f'Fold{sf-1}_D{ds}_Model_01-'
    name = ''
    for file in  os.listdir(path+'\\Folds'):
        #print(pref, '-',file,'-', os.path.isfile(os.path.join(path,'Folds\\'+file)), pref in file)
        if ( os.path.isfile(os.path.join(path,'Folds\\'+file)) ) and (pref in file):
        #    print('\t\tHEY - '+file)
            name = file
    return f'{path}\\Folds\\{name}'

def predictModel(modelpath, inputs, verbose=0):
    model = mL.load_model(modelpath+'.h5')
    pred = model.predict(inputs, verbose = verbose)
    return pred

# ---------------------------------------- Start MAIN ----------------------------------------------------
def train(n_steps=11,n_epoch=100,batch_size=1024, n_targets=60, written=0, model_written=0
    ,folded=0, downsample=1, modelname='test' , fname='testing', only_evaluate=0, usf=0, startfold = 1, description = None, withEval = False, withFold = False):


    #print(type(folded), len(folded), folded)
    #print('Folded: ', folded)
    
    descr = description
    #ds = mL.pDataset(fil_Name=DB_name , description = descr) 
    #       Loading Input
    m_input = t.f.root.all.pos_window
    #       Loading Input Labels    
    target = t.f.root.all.pos_target_window
    #       Loading Meta data
    m_meta = t.f.root.all.meta_window

    nvideos = 60
    nversions= 48
    n_splits = 1
    #begining = ds.beg

    PNR_fname = 'TXT/PersonNReg.txt'
    with open(PNR_fname, 'rb') as fp:   # Unpickling
        personNReg = mL.pickle.load(fp)
    
    lengths = mL.np.zeros((len(personNReg),))
    for n in range(len(personNReg)):
        b = len(personNReg[n]) % downsample !=0
        lengths[n] = len(personNReg[n])//downsample + b
        #lengths[n] = len(personNReg[n])#//downsample + b

    #           Build input variables
    n_samples = len(m_input)
    n_features = m_input.shape[2]
  


    #print(only_evaluate, usf)
    print('\t ---------------- Starting building ---------------- ')
    start = mL.t.time_ns()

    inp_path, tar_path, che_path, par_path, fold_path= buildModelWorkspace(n_features, fname, description = descr)
    iname = f'Inputs/{buildInputName( n_features, n_steps, downsample,description = descr)}'
    tname = f'Inputs/{buildTargetName( n_features, n_steps, downsample,description = descr)}'
    
    #          Training Variables
    n_samples , n_steps, n_feat = m_input.shape
    mname= f'{par_path}\\{modelname}'

    #         print Dimensions info

    print('\tInput shape:', m_input.shape, '\tTarget shape',target.shape)
    print('\t n_samples: ',n_samples,' |\tn_steps: ', n_steps,' |\tn_features: ', n_feat)
    print(f'\tModel Name : {mname}')
    print('\n\tPress Enter to continue ...')

    print('\t ---------------- Starting Training ---------------- ')
    
    process = psutil.Process(os.getpid())
    
    Fold_path = inp_path
    
    res_train = t.getCV_split(m_meta)

    print('test_set')
    print(len(res_train[1]))
    print('train_set')
    print(len(res_train[0]))

    memory = mL.np.array([process.memory_info().rss/(1024*1024)])
    print('Memory occupied:  ' + f'{memory[0]:.2f}' + ' MB')            
    
    if not model_written:
        model = buildModel(n_feat, n_steps, n_targets, modelname)
    else:
        model = mL.load_model(mname+'.h5')

    model.summary()
    #mL.tf.keras.utils.plot_model(model,to_file= f'{mname}.png',  show_shapes=True)
    
    ftrain_start = mL.t.time_ns()

    if not only_evaluate:
        train_set = res_train[0]
        test_set = res_train[1]
        train_perc, test_perc = len(train_set) / m_input.shape[0], len(test_set) / m_input.shape[0]
                    
        print(f'Train: {train_perc*100:.2f}%\t Test: {test_perc*100:.2f}%')
       
        model = train_model(train_set, test_set, m_input, target, 1, che_path, par_path, n_epoch, batch_size, mname, process, model, downsample, withFold=withFold)

    print('\t ---------------- Starting Predicting ---------------- ')

    vAL.VizAccLoss1(f'{par_path}\\', folder = 'Checkpoints\\', name = f'final_history')

    val_set = res_train[1]
    #predict = model.predict(m_input[val_set,:,:], verbose = 1)
    eval_data= My_Validation_Generator(val_set, m_input, batch_size)
    
    #mname= f'{par_path}/Checkpoints/final_Model_17-0.142'
    predict = predictModel(mname, eval_data, verbose = 1)

    i = mL.np.argmax(predict ,axis = 1)
    #print('predict :\t',type(predict),predict.shape)
    #print('index of max predict: \t', type(i), i.shape)
    #print('Expected from predict:\t', type( target[val_set]), target[val_set].shape )

    printConfusionMatrix(target[val_set],i)
    #printHeatMap(predict)


    #            Load Label Info
    #labels = mL.pd.read_json('labels_detailed1.json', orient='split')
    #print(len(target[val_set]), len(i))
    #printOutLabels(labels,i, target[val_set])


    #               Graph for Predicted (red) vs Expected (green)

    #printScatter(test_target,i)

def evaluate(n_steps=11,n_epoch=100,batch_size=1024, n_targets=60, written=0, model_written=0
    ,folded=0, downsample=1, modelname='final_Model_33-0.320' , fname='jld_RawV', usf=0, startfold = 1, description = None):
    
    par_path = 'models/training_model/jld_RawV/Checkpoints/'
    modelname =  'final_Model_17-0.142' #'final_Model_02-0.036'
    mname= f'{par_path}{modelname}'

    m_input = t.f.root.all.jld_window
    target = t.f.root.all.jld_target_window

    res_train = t.getCV_split(t.f.root.all.meta_window)

    eval_set = res_train[1]
    print(m_input.shape)
    eval_data= mL.tf.data.Dataset.from_generator(lambda: My_Validation_Generator (eval_set, m_input, batch_size), (mL.tf.float32,), output_shapes = ((None, m_input.shape[1], m_input.shape[2]),) )


    pred = predictModel(mname, eval_data, verbose = 1)
    print(pred)

    i = mL.np.argmax(pred ,axis = 1)
    printConfusionMatrix(target[eval_set],i)
    
    return pred, target, eval_set

def test_generators(train_set, test_set, m_input=t.f.root.all.pos_window, target=t.f.root.all.pos_target_window, batch_size=1024, ind = 0):
    train_ds = mL.tf.data.Dataset.from_generator(lambda: My_Input_Generator(train_set, m_input, target, batch_size),(mL.tf.float32,mL.tf.float32), output_shapes = ((None,m_input.shape[1],m_input.shape[2]),(None,)))
    test_ds = mL.tf.data.Dataset.from_generator(lambda: My_Input_Generator (test_set, m_input, target, batch_size), (mL.tf.float32,mL.tf.float32), output_shapes = ((None,m_input.shape[1],m_input.shape[2]),(None,)) )

    for i,n in enumerate(test_ds):
        if i == ind:
            print(i, n[0], n[1])
            return n[0]
            #print(n.shape)
            input()


    return train_ds, test_ds
model = PA.PLSTM()