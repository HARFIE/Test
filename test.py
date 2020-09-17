import myLib as mL
import tables
import csv
import numpy.linalg as la

INPUTPATH = 'Inputs\\'
DBNAME = 'NPY\\DataBlock5.npy'
jsonFoldersLoc = 'C:\\Users\\up201\\Desktop\\Tese\\openposeOriginal\\openpose-master\\examples\\media\\json3\\'
Edge_joints = [0,4,7,11,14]

def loadTarget():
	return mL.np.load(INPUTPATH+'target.npy')

def getsplit():
	resTrain = mL.getSplitSetsBasic4((3,8,2,60), 1, [6,7])
	return resTrain

def printScatters(test, train):
	mL.plt.scatter(test, test, label = 'test')
	mL.plt.scatter(train, train, label = 'train')
	mL.plt.xlabel('Video_Samples')
	mL.plt.ylabel('Video_Samples')
	mL.plt.title('Train - Test Split')
	mL.plt.show()

def getOldSplit():
	A= mL.np.arange(2880).reshape((60,48))

	r,p,c = 0,0,0
	v = []
	for a in range(60):
		v+= [a%48]

	return v 
def printOldSplit(v):
	for n in range(48):
		mL.plt.scatter(range(60), [n]*60, color='blue')
	mL.plt.scatter(range(60), v, color = 'orange')
	mL.plt.title('Train Test Split')
	mL.plt.xlabel('Action Labels')
	mL.plt.ylabel('Versions (Camera, Subject, Hat)')
	mL.plt.show()

def countErrorsJOD(inp):
	inp2 = inp.reshape((inp.shape[0], 15, 14,3))

	b = mL.np.count_nonzero(inp2[:,:,:,2] == 0, axis = 3)
	return b

def run(angle):
	
	a = evaluationHist('Results')
	a.draw(angle, 4)
	return 1

class myGraph():
	def __init__(self,name):
		self.name = name

class lengthHist(myGraph):
	def __init__(self,name,shapes, sumby=0):
		super().__init__(name)
		
		lens = mL.buildLengths(1)
		org =lens.reshape(shapes)
		sums = org.sum(axis = sumby)
		print(sums.shape)

		self.y= sums
	def __getitem__(self,index):
		#print(self.y,index)
		return self.y[index]
	def draw(self,):
		mL.plt.bar(range(1,self.y.shape[0]+1 ), self.y)
		mL.plt.title('Jsons by Action')
		mL.plt.xlabel('Action')
		mL.plt.ylabel('Jsons(Samples)')
		mL.plt.show()

class evaluationHist(myGraph):
	def __init__(self, name, resultsPath= 'models\\EvalRes.csv'):
		super().__init__(name)
		res = []
		with open(resultsPath, 'r') as file:
			reader = csv.reader(file, delimiter=',')
			for row in reader:
				res += row
		self.loss= [float(n) for i,n in enumerate(res) if (i)%3==0]
		self.acc= [float(n) for i,n in enumerate(res) if (i-1)%3==0]
		self.names = [(n) for i,n in enumerate(res) if (i+1)%3==0]

	def draw(self,angle = 'vertical', divideby = 1):
		barsLen = len(self.names)/divideby
		#print(barsLen, int(barsLen))
		colorsA = mL.np.array([0,0,0.25])
		colorsB = mL.np.array([1, 0.25, 0])
		color_step  = 0.75/barsLen
		barsLen = int(barsLen)
		for n in range(0,len(self.names), barsLen):	
			print(f'')
			axL=mL.np.arange(len(self.loss[n:n+barsLen]))+n
			barL = self.loss[n:n+barsLen]
			cL = tuple(colorsB)

			axA=mL.np.arange(len(self.acc[n:n+barsLen]))+n+0.25
			barA = self.acc[n:n+barsLen]
			cA = tuple(colorsA)
			
			print(f'len Loss: {len(axL)}, {len(barL)}, {len(cL)}')
			print(f'len Acc: {len(axA)}, {len(barA)}, {len(cA)}')
			mL.plt.bar(axL,  barL, width = 0.5, label = 'loss', color = cL  )
			mL.plt.bar(axA,  barA, width = 0.5, label = 'acc', color = cA  )

			colorsA[2] +=color_step
			colorsA[1] += color_step/2
			colorsA[0] += color_step/2

			colorsA= mL.np.roll(colorsA,1)

			colorsB[2] += color_step/2
			colorsB[0] -= color_step/2
			colorsB[1] += color_step
			
		ax = mL.plt.gca()
		mL.plt.title(self.name)
		ax.set_xticks(mL.np.arange(len(self.names)))
		ax.set_xticklabels(self.names, rotation=angle)
		ax.legend()
		mL.plt.tight_layout()
		mL.plt.show()
		#mL.plt.bar(mL.np.arange(len(self.loss)),self.loss, width = 0.5, label = 'Loss')
		
		#mL.plt.bar(mL.np.arange(len(self.acc))+0.25,self.acc, width = 0.5, label = 'Acc')
		


class fullDataset():
	def __init__(self, dataset_path):
		self.home_folder =folderDescr(dataset_path)
	def __getitem__(self,index):
		return self.home_folder[index]
	def __len__ (self,):
		return len(self.home_folder) 
	
	def getLens(self,):
		self.lengthsPerFolder = []
		for n in range(len(self.home_folder)):
			self.lengthsPerFolder += [len(self.home_folder[n][0].contentName)]
		return self.lengthsPerFolder	
	
	def getFolderNames(self,):
		return self.home_folder.contentFoldersName
	def showLens(self,):
		l = self.getLens()
		f = self.getFolderNames()

		mL.plt.bar(range(len(l)), l )
		ax = mL.plt.gca()
		mL.plt.title('Folder Lengths')
		ax.set_xticks(range(len(l)))
		ax.set_xticklabels(f, rotation = 'vertical')
		ax.set_ylabel('Number of videos')
		mL.plt.tight_layout()
		mL.plt.show()

	def getVideoNames(self,):
		n_setups = self.home_folder.inFolderNum
		setups = [None] * n_setups
		for n in range(n_setups):
			for m in range(self.home_folder.inFolderLens[n]):
				print(self.home_folder.inFolders[n][0].home_dir)
				setups[n] =  self.home_folder.inFolders[n][0].contentName

		return setups
	def getSubjectsDistribution(self,videoNames):
		subjects = videoNames.copy()
		for n in range(len(subjects)):
			videos= subjects[n]
			for m in range(len(videos)):
				subjects[n][m] =  int(videos[m][9:9+3])
			subjects[n] = mL.np.array(subjects[n]).astype(int)

		return subjects
	def showSubjectsDistributionPerSetup(self, subDist):
		subDist_u = subDist.copy()
		s = set()
		for n in range(len(subDist)):
			subDist_u[n] = mL.np.unique(subDist[n])
			s |= set(subDist_u[n])


		sub_heatmap = mL.np.zeros((len(subDist), len(s)))
		print(sub_heatmap.shape)
		for n in range(sub_heatmap.shape[0]):
			for m in subDist_u[n]:
				sub_heatmap[n][m-1] = 1

		mL.sns.heatmap(sub_heatmap, cmap='Blues', linewidths = 0.1, xticklabels = range(1,sub_heatmap.shape[1]+1), yticklabels = range(1,sub_heatmap.shape[0]+1))
		ax = mL.plt.gca()
		ax.set_xlabel('Subjects')
		ax.set_ylabel('Setups')
		mL.plt.title('Subject Distribution along Setups')

		mL.plt.show()
		return subDist_u

class folderDescr():
	def __init__(self,dataset_path):
		self.home_dir= dataset_path
		print(f'Initiating folderDescr: {dataset_path}')
		if not mL.os.path.isdir(dataset_path):
			print(f'Not a Path: {dataset_path}')
			return -1

		self.contentName = mL.os.listdir(dataset_path)
		self.contentFoldersName = [n for n in self.contentName if mL.os.path.isdir(f'{dataset_path}\\{n}')]
		self.inFolderNum = len(self.contentFoldersName)
		folder_paths = [f'{dataset_path}\\{n}' for n in self.contentFoldersName]
		self.inFolders = [None]*len(folder_paths)
		self.inFolderLens = [None]*len(folder_paths)

		for i,n in enumerate(folder_paths):
			self.inFolders[i] = folderDescr(n)

			self.inFolderLens[i] = len(self.inFolders[i])
		
	def __getitem__ (self, index):
		return  self.inFolders[index]

	def __len__(self,)	:
		return len(self.contentFoldersName)

def createDataset():
	f = tables.open_file('Tables/Dataset.h5', 'a')
	for n in range(17):
		f.create_group('/', f'setup{n+1:02}', f'Setup {n+1}')
	f.close()

def loadDataset():
	return tables.open_file('Tables/Dataset.h5', 'a')
class Pos(tables.IsDescription):
	pos=tables.FloatCol(shape=(30,))
	vname =  tables.StringCol(20)
	jno = tables.IntCol()
	pco = tables.IntCol()

class Metadata(tables.IsDescription):
	jno = tables.IntCol()
	pco = tables.IntCol()

	setup = tables.IntCol()
	camera = tables.IntCol()
	person = tables.IntCol()
	replication = tables.IntCol()
	action = tables.IntCol()

class Sample(tables.IsDescription):
	vname =  tables.StringCol(20)
	jno = tables.IntCol()
	pco = tables.IntCol()
	
	class features(tables.IsDescription):
		pos = tables.FloatCol(shape=(30,))
		jod = tables.FloatCol(shape=(630,))
		jld = tables.FloatCol(shape=(364,))
		lla = tables.FloatCol(shape=(756,))
	#features =  tables.FloatCol(shape=(3,))

def addTable(f,group, tablename='rawFeatures',description = Sample):
	f.create_table(group, tablename, description)
	return f
def addTables(f,tablename = 'rawFeatures'):
	for n in range(17):
		group = f'/setup{n+1:02}'
		f.create_table(group, tablename, Sample)
	return f
def fillRandomRow(table):
	row = table.row

def parseVideoName(videoName):
	setup = int(videoName[1 : 1+3])
	camera = int(videoName[5 : 5+3])
	person = int(videoName[9 : 9+3])
	replication = int(videoName[13 : 13+3])
	action = int(videoName[17 : 17+3])

	return setup, camera, person, replication, action

def readJson(name, verbose = 0):
	with open(name) as f:
		data = mL.json.load(f)

	pco = len(data['people'])
	if verbose:
		print('readJson:\n', pco)
	if pco >0:
		pose_data = data['people'][0]['pose_keypoints_2d']

		xCoordinates = [n for i,n in enumerate(pose_data) if (i%3 == 0) and (i//3 <15)]
		yCoordinates = [n for i,n in enumerate(pose_data) if ((i+2)%3 == 0) and (i//3 <15)]

		jno = int(name[-15-12:-15])

		return (xCoordinates, yCoordinates, jno, pco)
	else:
		return None

def readjsonFolder(jsonFolderpath, verbose = 0):
	dirlist = mL.os.listdir(jsonFolderpath)
	nfiles = len(dirlist)
	count = 0
	x_arr, y_arr, j_arr, pc_arr = [None]*nfiles,[None]*nfiles,[None]*nfiles, [None]*nfiles
	for i,n in enumerate(dirlist):
		test = readJson(jsonFolderpath+n, verbose = verbose)
		if test:
			x_arr[i-count], y_arr[i-count], j_arr[i-count] , pc_arr[i-count] = test
		else:
			count +=1
		if verbose:
			print('readjsonFolder:\n', test)
	if count :
		return (x_arr[:-count], y_arr[:-count], j_arr[:-count], pc_arr[:-count])
	else:
		return (x_arr, y_arr, j_arr, pc_arr)
def JsonReader(f):
	dirlist = mL.os.listdir(jsonFoldersLoc)
	videos = [None]*len(dirlist)
	begining = 0
	prev_value = 0
	same_flag = 0
	groups = f.list_nodes('/')
	start =mL.t.time_ns()
	verbose = 0
	flag = 0
	oldS = 0
	oldC = 0
	oldP = 0

	nextSet = 0#2
	nextCam = 0#2
	nextPer = 0#9
	nextRep = 0#2
	nextAct = 0#17

	finish = 0
	endAct = 61# 30
	aux_c = 0
	aux_c2=0
	durJ=0
	startS = mL.t.time_ns()
	startC = mL.t.time_ns()
	startP = mL.t.time_ns()
	for i,n in enumerate(dirlist):
		s,c,p, r,a = parseVideoName(n)

		if (s>=nextSet) and (c>=nextCam) and (p>=nextPer) and (r>=nextRep) and (a>=nextAct) and not finish:
			verbose = 0
			flag = 1
			if a==endAct:
				finish +=1
				flag =0

		else:
			verbose =0
			flag = 0
		if flag:
			videos[i] = readjsonFolder(jsonFoldersLoc+ n+ '\\', verbose =verbose)
			aux_c +=1
			aux_c2 += len(videos[i][0])
			#print(n,aux_c2,aux_c)
			#for k in range(len(groups)):
			#print(len(groups), groups[s-1+3])
			#input()
			savePosRows(f.root.all.tAll, videos[i][0], videos[i][1], videos[i][2], videos[i][3], n, verbose = verbose)
		
		if p!= oldP:
			durP =mL.t.time_ns() - startP
			if i!=0:
				print(f'\t\t\t\tSubject {oldP}', mL.buildTime(mL.ns2ms(durP)), 'ms')
				f.flush()
			oldP = p
			startP =mL.t.time_ns()
	
		if c != oldC:
			durC =mL.t.time_ns() - startC
			if i!=0:
				print(f'\t\tCamera {oldC}: ', mL.buildTime(mL.ns2ms(durC)), 'ms')
			oldC = c
			startC =mL.t.time_ns()

		if s != oldS:
			durS =mL.t.time_ns() - startS
			if i!=0:
				print(f'Setup {oldS}: ', mL.buildTime(mL.ns2ms(durS)), 'ms')
			oldS = s
			startS =mL.t.time_ns()
		
	f.flush()
	print(len(f.root.all.tAll))
	dur = mL.t.time_ns() - start
	print(mL.buildTime(mL.ns2ms(dur)), 'ms')

	return videos, dirlist

def savePosRows(table, x, y, j, pc,name, verbose =0):
	row = table.row
	Pos = mL.np.zeros(row['features/pos'].shape)
	for x_js, y_js, jno, pc_js in zip(x,y,j, pc):
		if verbose:
			print('SavePosRows:\n',x_js, y_js, jno, pc_js, name)
		Pos[0:30:2]=x_js
		Pos[1:30:2]=y_js
		row['features/pos']= Pos
		
		row['jno'] = jno
		row['vname'] = name		
		row['pco'] = pc_js
		row.append()
		#print(len(table))
	table.flush()

def copyTables(t1, t2):
	fields = t1.coldescrs.keys()
	for row in t1.iterrows():
		r= t2.row
		for f in fields:
			r[f]=row[f]
		#r['date']= ' '
		r.append()
	t2.flush()


def buildMeta(t_source, t_destination):
	count = 0
	for row in t_source.iterrows():
		r= t_destination.row
		r['pco'] = row['pco']
		r['jno'] = row['jno']
		name = row['vname']
		r['setup'] = int(name[1:4])
		r['camera'] = int(name[5:8])
		r['person'] = int(name[9:12])
		r['replication'] = int(name[13:16])
		r['action'] = int(name[17:20])

		r.append()
		count+=1
		if count % 100000 == 0:
			print(count, '/', len(t_source))
	t_destination.flush()	
def buildWindowMeta(t_source, t_destination, n_steps = 11):
	counter = 0
	for row in t_source.iterrows():
		if row['jno']>=n_steps-1:
			r= t_destination.row
			r['pco'] = row['pco']
			r['jno'] = row['jno']

			name = row['vname']
			r['setup'] = int(name[1:4])
			r['camera'] = int(name[5:8])
			r['person'] = int(name[9:12])
			r['replication'] = int(name[13:16])
			r['action'] = int(name[17:20])

			r.append()
			counter +=1

		if (counter+1) % 10000 == 0 :
			print('----------------------------------------------------------------')
			print(row)
			print(r)
			print(f'counter {counter} ')

	t_destination.flush()	

def getCS_split(meta):
	All = set(range(len(meta)))

	condition = ''
	for i,n in enumerate(CS_test):
		if i==0:
			condition += f'(person == {n})'
		else:
			condition += f' | (person == {n})'

	print(condition	)
	test = meta.get_where_list(condition)

	train = All - set(test)

	return list(train), list(test)

def getCV_split(meta):
	All = set(range(len(meta)))

	#condition = ''
	#condition += f'(setup == {1})'
	#All = set( meta.get_where_list(condition).astype(int) )

	condition = ''
	for i,n in enumerate(CV_test):
		if i==0:
			condition += f'(camera == {n})'
			#condition += f'((setup == {1}) & (camera == {n}))'
		else:
			condition += f' | (camera == {n})'
			#condition += f' | ((setup == {1}) & (camera == {n}))'

	print(condition	)
	test = meta.get_where_list(condition)
	test = test.astype(int)
	train = All - set(test)

	return list(train), list(test.astype(int))


def window_stack(a, stepsize=1, width=3):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )
    
def buildWindow(All, array, length=30, path = 'features/pos', n_steps = 11):
	counter = 0
	w_counter = 0
	window = mL.np.zeros((n_steps ,length)).astype(float)
	for original in All.iterrows():
		#if counter > 4230000 : 
		#	print(counter)
		if original['jno']< n_steps:
			if original['jno'] == 0:
				w_counter = 0
			window[w_counter] = original[path]
			#print(window[w_counter])
			#input()
			w_counter += 1
			if original['jno'] == n_steps-1:
				#print(window[:,:5])
				#print('INIT')
				#input()
				array[counter,:,:] = window
				counter += 1

		else:
			window = mL.np.roll(window, -1, axis = 0)
			window[-1] = original[path]

			#print(window[:,:5])
			#print('FIN')
			#input()
			array[counter,:,:] = window
			counter += 1

		if (counter+1) % 10000 == 0 :
			print('----------------------------------------------------------------')
			#print(original)
			#print(window)
			print(array)
			print(f'{counter+1} / {len(array) }')
			#input()
	print(counter)

def createCarray(file,path, name, shape , atom= tables.FloatAtom()):
	file.create_carray(path, name, atom, shape)

			
#addTables(f)

def splitVname(name):
	setup = int(name[1:4])
	camera = int(name[5:8])
	person = int(name[9:12])
	replication = int(name[13:16])
	action = int(name[17:20])

	return setup, camera, person, replication, action

def buildTargetWindow(All, arr, n_steps = 11):
	counter = 0
	for row in All.iterrows():
		s,c,p,r,a = splitVname(row['vname'])
		
		if row['jno'] >= n_steps-1:
			arr[counter] = a-1
			counter += 1

		if (counter+1) % 10000 == 0 :
			print('----------------------------------------------------------------')
			print('row:', row)
			print('window: ', arr)
			print(f'counter {counter+1} / {len(arr)}')
	return counter
 
def getBegZeros(table):
	beg_ind = []
	all_ind = []
	for i,row in enumerate(table.iterrows()):
		#print(row)
		#print(row['features/pos'])
		#input()
		
		if row['jno'] == 0:
			Bseq = 1

		if (mL.np.any(row['features/pos'] == 0)):
			all_ind += [i]
			if (Bseq):
				beg_ind +=[i]

				#print(i, beg_ind)
				#input()
		else: 
			Bseq = 0

		if (i +1) % 100000 == 0:
			print(f'{i+1:07} / {len(table)}')

	return beg_ind, all_ind
def getEndZeros(table):
	end_ind = []
	Eseq = 1
	last_ini = 0
	counter = 0
	for i in range(len(table)-1,-1,-1):
		row = table[i]
		#print(type(row))
		if (Eseq) and (mL.np.any(row['features']['pos'] == 0)):
			end_ind += [i]
			counter += 1
			#if i+1 <len(table)-1:
			#	print(table[i+1]['jno'])
			#print(i, row['jno'])
		else: 
			Eseq = 0
		
		if row['jno'] == 0:
			Eseq = 1
			#print(end_ind[last_ini:])
			last_ini = counter
			#input()


		if (i +1) % 100000 == 0:
			print(f'{i+1:07} / {len(table)}')
	
	return end_ind

def filter(table):
	window = [0]*10
	counter = 0

	start = mL.t.time_ns()
	beg_ind, all_ind= getBegZeros(table)
	beg_ind = set(beg_ind)
	all_ind = set(all_ind)
	dur = mL.t.time_ns() - start
	print('Beg Duration: ', mL.buildTime(mL.ns2ms(dur)), 'ms')

	start = mL.t.time_ns()
	end_ind = set( getEndZeros(table) )
	dur = mL.t.time_ns() - start
	print('End Duration: ', mL.buildTime(mL.ns2ms(dur)), 'ms')

	print('all', len(all_ind), type(all))
	print('beg', len(beg_ind))
	print('end', len(end_ind))

	mid_ind = all_ind -beg_ind - end_ind
	print('mid', len(mid_ind))
	print('end  & beg', len(beg_ind-end_ind))

	return beg_ind, end_ind, mid_ind, all_ind

def applyFilter(table, mid):
	w_size = 10
	counter = 0
	start= mL.t.time_ns()
	for j,n in enumerate(mid):
		row = table[n]
		if row['jno']<10:
			w_size = row['jno']

		window = mL.np.zeros((w_size,))
		counter = 0
		a = mL.np.where(row['features']['pos'] == 0)
		a=a[0].astype(int)


		#print('n',n,'a', a,'w_size', w_size, 'j', j)
		#print(row['features']['pos'].shape)
		#print(row['features']['pos'], row['jno'])
		avg = mL.np.mean(table[n-w_size:n]['features']['pos'][:,a], axis = 0)

		aux = table.cols.features[n]
		aux['pos'][a] = avg
		table.cols.features[n] = aux
		#print(aux.shape, avg.shape, a.shape)
		#print(aux[a])
		#print(avg)
		#input()
		#row.update()
		if (j+1) % 100000 == 0:
			print(f'{j+1:06} / {len(mid)}')
		#print(avg, '\n--------------------------')
	#table.flush()
	dur = mL.t.time_ns()-start
	print(f'Duration: {mL.buildTime(mL.ns2ms(dur))} ms')

def testFneig(Gr, source, dest):
    b= list(mL.nx.all_neighbors(Gr, source))
    if dest in b:
        return True
    else:
        return False

def testEbody(Gr, source, dest):
    sE = source in Edge_joints
    dE =  dest in Edge_joints
    sdE = source,dest in Edge_joints
    ignore = [x in [source,dest] for x in [15, 16, 19, 20, 21, 22, 23, 24]]
    if( True in ignore  ):
            #print('ignore')
            return False
    if not (sE  or dE):
        return False
    elif(sE*dE ):
        return False

    
    a_exp = 2
    if(0 in [source,dest]):
        a_exp=1
        

    a = mL.nx.shortest_path_length(Gr, source , dest)   
    

    if ( (a != a_exp )):
        return False
    else:
        return True

def testBedges(Gr, source, dest):
    sE = source in Edge_joints
    dE =  dest in Edge_joints
    sdE = (source,dest) in Edge_joints
    if (sE) and(dE):
        #print('sdE:', sdE)
        return True
    else:
        return False

def buildLinIdx(G):

    li = []
    li_tfn = []
    li_teb = []
    li_tbe =[]

    for s in range(15):
        for d in range(15):
            # test joints first neighbor
            if(s==d):
                tfn, teb, tbe = False, False, False
            else:
                tfn= testFneig(G,s,d)
                if(tfn):
                    if not ([d,s] in li_tfn):
                        li_tfn.append([s,d])

                # test one joint on the edge and othe is at max distance of two
                teb= testEbody(G,s,d)
                if(teb):
                    if not ([d,s] in li_teb):
                        li_teb.append([s,d])
                    
                
                # test both joint on edges
                tbe = testBedges(G,s,d)
                if(tbe):
                    if not ([d,s] in li_tbe):
                        li_tbe.append([s,d])
                   
                # Compute distance
            if (tbe) or (tfn) or (teb):
                if not ([d,s] in li):
                    li.append([s,d])

    return li

def getHeight(P0x, P0y, P8x, P8y):
    heigth = mL.np.power(P0y - P8y,2) + mL.np.power(P0x - P8x,2) 
    heigth = mL.np.power(heigth, 0.5)
    return heigth

def getJLdesl(PL1, PL2, Pos, P):
	n=0
	dist = mL.np.zeros((Pos.shape[0]))
	#print('hey',Pos.shape)
	#input()

	heigth = getHeight(Pos[:,0], Pos[:,1],Pos[:,16],Pos[:,17])

	x0, y0 = Pos[: , 2*P]  ,  Pos[:, 2*P+1]
	x1, y1 = Pos[: , 2*PL1],  Pos[:, 2*PL1+1]
	x2, y2 = Pos[: , 2*PL2],  Pos[:, 2*PL2+1]
	num = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1  -  y2*x1
	num = abs(num)
	den = mL.np.power(y2-y1,2)+mL.np.power(x2-x1,2)
	den = mL.np.power(den, 0.5) * heigth

	res = num/den
	if (mL.np.any(den==0)):
		aux =list( mL.np.where(den ==0))[0]
		
		#print('x0:', x0.shape,'y0:', y0.shape, 'x1:', x1.shape,'y1:', y1.shape, 'x2:', x2.shape,'y2:', y2.shape)
		#print('x0:', x0[aux[:5]],'y0:', y0[aux[:5]], 'x1:', x1[aux[:5]],'y1:', y1[aux[:5]], 'x2:', x2[aux[:5]],'y2:', y2[aux[:5]])
		#print(len(num), num[aux[:5]])
		#print(len(den), den[aux[:5]])
		#print(len(res), res[aux[:5]])
		#print(len(aux), aux[:5])
		#input()
		#print(res)
		
		res[aux]= -1
		#print(f'hey There is -1 here: {PL1}--{PL2} ..... {P}' )
		#print(mL.np.any(den ==0), len(aux), y2[aux[0]], y1[aux[0]], x2[aux[0]], x1[aux[0]], num[aux[0]])
	if (mL.np.any(mL.np.isnan(res))):
		print('hey There is NANs')

	dist[:] = res

	#print(dist.shape)
	return dist

def JLD(pos, lins ):
	counter = 0
	D = mL.np.zeros((pos.shape[0],364 ))
	#print(pos.shape)
	for p in range(15):
		for l in range(lins.shape[0]):
			if (not p in lins[l,:]):	
				D[:, counter]= getJLdesl(lins[l,0], lins[l,1], pos, p)
				counter +=1
		#print(p+1, '/' ,  15, f' ------ counter: {counter}')
	return D

def buildJLD(All):
	Gra=mL.build_graph()			# Graph of skeleton to extract JLD feature
	lines = mL.np.array(buildLinIdx(Gra))
	start = mL.t.time_ns()
	b_start = mL.t.time_ns()
	last_end = 0

	for n in range(len(All)//100000):
		if n >=37:
			a=JLD(All.cols.features.pos[last_end:last_end+100000], lines)
			All.cols.features.jld[last_end:last_end+100000]=a
			last_end+=100000
			if (n +1) % 2 == 0 :
				print(a.shape)
				dur = mL.t.time_ns()-start
				b_dur = mL.t.time_ns()-b_start
				print(f'{n+1} / {len(All)//100000}, Batch Duration: {mL.buildTime(mL.ns2ms(b_dur))} ms, Running Time: {mL.buildTime(mL.ns2ms(dur))} ms ')
				b_start = mL.t.time_ns()
		else:
			last_end= (n+1)*100000
	a=JLD(All.cols.features.pos[last_end:],lines)
	All.cols.features.jld[last_end:]=a
	last_end+=100000

	dur = mL.t.time_ns()-start
	print(f'Full Duration: {mL.buildTime(mL.ns2ms(dur))} ms')


def getJdesl(pos, j1, j2):
    #print('getJdesl', pos.shape)
    ori = mL.np.zeros((pos.shape[0], 2)) # (n, 2)
    dist = mL.np.zeros((pos.shape[0], 1))  # (n, 1)
    
    x, y = pos[:, 2*j1]-pos[:, 2*j2], pos[:, 2*j1+1]-pos[:, 2*j2+1]

    vector = mL.np.array([x,y])
    norm = la.norm(vector, axis=0)

    dist[:,0]= norm #[heigth_nonzero] #/ heigth[heigth_nonzero]


    no = mL.np.where(norm !=0)[0]
    if no.size > 0:

        aux = mL.np.transpose( vector[:,no]/norm[no])
        ori[no, :] =  aux       #2*(s*24+d):2*(s*24+d)+2
    
    return ori, dist

def JOD( pos):
    print('Generating JOD')

    j1=0
    j2=1
    col_name=[]
    oriG = mL.np.zeros((pos.shape[0], 15*15-15, 2))
    distG = mL.np.zeros((pos.shape[0], 15*15-15, 1))
    counter = 0

    for j1 in range(15):
        #print(j1,'---------------------')
        for j2 in range(15):
            if j1 != j2:
                ori, dist =getJdesl(pos, j1, j2) # (n, 2) , (n,1)

                oriG[:,  counter]= ori
                distG[:, counter]= dist

                counter +=1
            if (counter+1)%20 == 0:
            	print(counter+1, '/' , 15*15)

    #print('oriG', oriG.shape, 'distG',distG.shape)

    gather = mL.np.hstack((oriG.reshape(pos.shape[0], oriG.shape[1]*oriG.shape[2]),distG.reshape(pos.shape[0], distG.shape[1]*distG.shape[2])))
    #print('ori', ori.shape,'\tdist', dist.shape, '\toriG', oriG.shape, '\tdistG', distG.shape)
    return gather

def buildJOD(All):
	last_end=0
	for n in range(len(All)//100000):
		a=JOD(All.cols.features.pos[last_end:last_end+100000])
		All.cols.features.jod[last_end:last_end+100000]=a
		last_end+=100000
		if (n +1) % 2==0 :
			print(f'{n+1} / {len(All)//100000}')
	a=JOD(All.cols.features.pos[last_end:])
	All.cols.features.jod[last_end:]=a
	last_end+=100000
	
	print(a.shape)



def LLA( pos):
    print('Generating LLA')
    Gra=mL.build_graph()
    lines = mL.np.array(buildLinIdx(Gra))
    Angles = mL.np.zeros((pos.shape[0], lines.shape[0]*lines.shape[0] - lines.shape[0]) )
    counter = 0
    print(Angles.shape)
    l1c=0
    l2c=0
    cname = []

    window= 10
    w_sum =0
    w_count=0
    for line_1 in lines:
        for line_2 in lines:
            if not mL.np.array_equal(line_1 , line_2):

                cname.append( f'{line_1}--{line_2} angle' )
                P1 = line_1[0]
                P2 = line_1[1]
                P3 = line_2[0]
                P4 = line_2[1]
                x1,    y1=  pos[:, 2*P1], pos[:,2*P1+1]
                x2,    y2=  pos[:, 2*P2], pos[:,2*P2+1]
                x3,    y3=  pos[:, 2*P3], pos[:,2*P3+1]
                x4,    y4=  pos[:, 2*P4], pos[:,2*P4+1]
                dx21 = x2-x1
                dy21 = y2-y1
                dx43 = x4-x3
                dy43 = y4-y3
                inf21 = (dy21 == 0) * (dx21 == 0)
                inf43 = (dy43 == 0) * (dx43 == 0)
                ind21 = mL.np.where(inf21)[0]
                ind43 = mL.np.where(inf43)[0]

                if (mL.np.any(inf21)):

                    x2[ind21], y2[ind21] =   mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P2]) for i in ind21      ]), mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P2+1]) for i in ind21      ])
                    x1[ind21], y1[ind21] =   mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P1]) for i in ind21      ]), mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P1+1]) for i in ind21      ])
                    dx21 = x2-x1
                    dy21 = y2-y1
                
                if (mL.np.any(inf43)):
                    x4[ind43], y4[ind43] =   mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P4]) for i in ind43      ]), mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P4+1]) for i in ind43      ])
                    x3[ind43], y3[ind43] =   mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P3]) for i in ind43      ]), mL.np.array(  [mL.np.mean(pos[i-window:i, 2*P3+1]) for i in ind43      ])
                    dx43 = x4-x3
                    dy43 = y4-y3

                m_1 = (y2-y1) / (x2-x1 )
                m_2 = (y4-y3) / ( x4-x3)
                
                Angles[:, counter ] = mL.np.arctan((m_1*m_2)/(1+m_1*m_2))
                m1_inf =  (mL.np.isinf(m_1))
                m2_inf =  (mL.np.isinf(m_2))
                b_inf = m1_inf * m2_inf

                c_inf = mL.np.logical_and(mL.np.logical_not(b_inf), m1_inf)
                d_inf = mL.np.logical_and(mL.np.logical_not(b_inf), m2_inf)
                om1 = mL.np.where(c_inf)[0]
                #print(om1.shape)
                om2 = mL.np.where(d_inf)[0]
                #print(om2.shape)
                bm12 = mL.np.where(b_inf)[0]

                Angles[bm12, counter ] = mL.np.arctan(1)          
                if(om1.shape[0]>0):
                    Angles[om1, counter ] = mL.np.arctan(1)    
                if(om2.shape[0]>0):
                    Angles[om2, counter ] = mL.np.arctan(1)    


                if mL.np.any(mL.np.isnan(Angles)):
                    ind =mL.np.where(mL.np.isnan(Angles))[0]
                    Angles[ind, counter ]= 10
                
                if mL.np.any(mL.np.isnan(Angles)):
                    ind =mL.np.where(mL.np.isnan(Angles))[0]
                    
                    print('\n\t',len(ind), '\t',Angles[ind[0], counter ])
                    print(f'm_1: {m_1[ind[0]]}, m_2: {m_2[ind[0]]}')
                    print('\t', m_1[ind[0]] * m_2[ind[0]], '/' ,  1+ m_1[ind[0]]*m_2[ind[0]],'=',  m_1[ind[0]] * m_2[ind[0]] /  1+ m_1[ind[0]]*m_2[ind[0]]  ,'\n')

                if ( counter%10 ==0):
                    print('\t ', counter, '/', lines.shape[0]*lines.shape[0]-lines.shape[0])
                l2c += 1
                counter +=1
        l1c += 1
        l2c=0
    l1c=0
    l2c=0
    return Angles


def buildLLA(All):
	Gra=mL.build_graph()
	lines = mL.np.array(buildLinIdx(Gra))
	last_end=0
	for n in range(len(All)//100000):
		a=LLA(All.cols.features.pos[last_end:last_end+100000])
		All.cols.features.lla[last_end:last_end+100000]=a
		last_end+=100000
		if (n +1) % 1==0 :
			print(f'{n+1} / {len(All)//100000}')
	a=LLA(All.cols.features.pos[last_end:])
	All.cols.features.lla[last_end:]=a
	last_end+=100000
	
	print(a.shape)


def my_modify_rows(All, source_start, source_stop, dest_start, dest_stop, source_step = 1, batch_size = 1000000): 
	#pick source range and shift it to dest range
	no_shf= mL.np.ceil(-( dest_start -source_start ) / (source_stop - source_start)).astype(int) # dest is upper than source
	print(no_shf, leng)
	input()

	for n in range(no_shf-1):
		leng = source_stop-source_start
		off_set = (n+1)*leng
		
		#shift up
		print(f'{source_start - off_set} : { source_stop - off_set} ---> Aux')
		aux = All[source_start - off_set: source_stop - off_set]

		print(f'{source_start - n*leng} : { source_stop - n*leng} ---> {source_start - off_set} : { source_stop - off_set}')
		All.modify_rows(source_start - off_set, source_stop - off_set , source_step, All[source_start - n*leng: source_stop - n*leng])

		print(f'Aux ---> {source_start - n*leng} : { source_stop - n*leng}')
		All.modify_rows(source_start - n*leng, source_stop- n *leng , source_step, aux)

		#print(All[:])
		print(f'--------------------------------------------------- {n+1} / {no_shf} -----------------------------------------------------')
	n+=1
	print(n, no_shf)

	print(f'{dest_start} : { dest_stop} ---> Aux')
	aux = All[dest_start : dest_stop]

	print(f'{source_start - n*leng} : { source_stop - n*leng} ---> {dest_start} : { dest_stop}')
	All.modify_rows(dest_start , dest_stop, source_step , All[ source_start - n*leng : source_stop - n*leng  ])

	print(f'Aux ---> {source_start - n*leng} : { source_stop - n*leng}')
	All.modify_rows(source_start - n*leng, source_stop- n *leng , source_step, aux)
	n+=1

	#print(All[:])
	print(f'----------------------------------------------------- {n+1} / {no_shf} -------------------------------------------------------')

CS_train = [ 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
CS_test = list(set(range(1,41))-set(CS_train) )
CV_train = [2,3]
CV_test = [1]

#createDataset()
f= loadDataset()