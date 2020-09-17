import myLib as mL
import matplotlib.colors as mcolors

PARENT_FOLDER_PATH = 'C:\\Chem\\chem\\MyScripts\\models\\training_model\\'
POS_PATH = 'model_POS_f31\\'
JOD_PATH = 'model_JOD_f631\\'
JLD_PATH = 'model_JLD_f365\\'
LLA_PATH = 'model_LLA_f757\\'

background_color = [0.9,0.9,0.9]
def printAcc( hist_list, colorsR, colorsE, labels, figpath):
    print('Printing Accuracy graph ...\n')

    # Acc
    for n in range(len(hist_list)):
        mL.plt.plot(hist_list[n][1:,2], c = colorsR[n], label= f'{labels[n]}_Train')
        mL.plt.plot(hist_list[n][1:,4], c = colorsE[n], label= f'{labels[n]}_Test')

    mL.plt.title('Model Accuracy')
    mL.plt.ylabel('Accuracy')
    mL.plt.xlabel('Epoch')
    mL.plt.legend( loc='lower right')
    mng = mL.plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    a=mL.plt.gca()
    a.set_facecolor(background_color)
    
    mL.plt.savefig(figpath+'AccGraph.png')
    mL.plt.show()

def printLoss(hist_list, colorsR, colorsE, labels, figpath):
    print('Printing Loss graph ...\n')
    
    for n in range(len(hist_list)):
        mL.plt.plot(hist_list[n][1:,1], c = colorsR[n], label= f'{labels[n]}_Train')
        mL.plt.plot(hist_list[n][1:,3], c = colorsE[n], label= f'{labels[n]}_Test')

    mL.plt.title('Model Loss')
    mL.plt.ylabel('Loss')
    mL.plt.xlabel('Epoch')
    mL.plt.legend( loc='lower right')
    a=mL.plt.gca()
    a.set_facecolor(background_color)
    mng = mL.plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    
    mL.plt.savefig(figpath+'LossGraph.png')
    mL.plt.show()

def printAcc1(hist, img_path):
    print('Printing Accuracy graph ...\n')

    mL.plt.plot(hist[1:,2], c= 'aquamarine', label = 'Train')
    mL.plt.plot(hist[1:,4], c= 'rosybrown', label = 'Test')


    mL.plt.title('Model Accuracy')
    mL.plt.ylabel('Accuracy')
    mL.plt.xlabel('Epoch')
    mL.plt.legend( loc='lower right')
    a=mL.plt.gca()
    a.set_facecolor(background_color)
    
    mL.plt.savefig(img_path+'AccGraph.png')
    mL.plt.show()

def printLoss1(hist, img_path):
    print('Printing Loss graph ...\n')
    
    mL.plt.plot(hist[1:,1], c= 'aquamarine', label = 'Train')
    mL.plt.plot(hist[1:,3], c= 'rosybrown', label = 'Test')

    mL.plt.title('Model Loss')
    mL.plt.ylabel('Loss')
    mL.plt.xlabel('Epoch')
    mL.plt.legend( loc='upper left')
    mL.plt.savefig(img_path+'LossGraph.png')
    a=mL.plt.gca()
    
    a.set_facecolor(background_color)
    mL.plt.show()

def buildRGB_Colors(component, N, cb,rev = False):
    w = [0,0,0,   0.5]
    colors = [None]*N

    for n in range(N):
        color = w.copy()

        baseWeights = mL.np.array([1,1,1])
        w1 = baseWeights
        w1[component] = 3
        if not rev:
            w1[(component+1)%3] = 0.5
        else:
            w1[(component-1)%3] = 0.5


        color1 = cb/ mL.np.sum(cb)
        color1 *= 0.5
        color1 += 0.5

        color1 -= 0.2*n 
#        color2 = (mL.np.random.rand(3)*0.5 + 0.25) * w1
        colors[n] = color1
    return colors

def vizAccLoss(par_fold_paths,son_dirName, son_filNames, labels,figpath):

    hist_paths = [None]* len(par_fold_paths)
    H = [None] * len(par_fold_paths)

    for n in range(len(par_fold_paths)):
        Dir    = par_fold_paths[n]
        subDir = son_dirName[n]
        name   = son_filNames[n]


        hist_paths[n] = f'{Dir}{subDir}{name}.csv'
        H[n] = mL.np.genfromtxt(hist_paths[n], delimiter = ',')

    rgb_comp = 1
    n_colors = len(par_fold_paths)
    colorBaseR, colorBaseE = mL.np.random.rand(3), mL.np.random.rand(3)
    colorsR = buildRGB_Colors(rgb_comp, n_colors, colorBaseR)
    colorsE = buildRGB_Colors((rgb_comp+1)%3, n_colors, colorBaseE,rev = True)

    printAcc(H, colorsR, colorsE, labels, figpath)
    printLoss(H, colorsR, colorsE, labels, figpath)

def VizAccLoss1(model_path, folder='Checkpoints\\', name = 'history'):
    hist_path = f'{model_path}{folder}{name}.csv'
    H = mL.np.genfromtxt(hist_path, delimiter= ',')

    printAcc1(H, model_path)
    printLoss1(H, model_path)

def resultsHistogram(descriptions ):
    return 0
if __name__ == '__main__':
	VizAccLoss1(f'{PARENT_FOLDER_PATH}{LLA_PATH}', folder = 'Folds\\', name = 'Fold_history')