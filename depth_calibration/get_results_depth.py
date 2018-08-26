import os, sys
from depth_helper import *
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
SHAPES_ROOT = os.getcwd().split("/silhouettes/")[0] + "/silhouettes/"
from keras.models import load_model
import keras.losses
import os.path
import glob


weights_path = '/home/mcube/weights_server_last/'
weights_files = glob.glob(weights_path + '*.hdf5')
weights_files.sort(key=os.path.getmtime)
#weights_files.sort()
train_loss = []; test_loss = []
std_train_loss = []; std_test_loss = []
it_count = 0
heights = []
for it, weights_file in enumerate(weights_files):
    '''
    if 'heigth' in weights_file: continue
    if '10_' in weights_file: continue
    if '50_' in weights_file: continue
    if '500' in weights_file: continue
    if '1000_' in weights_file: continue
    #if 'rgb' in weights_file: continue
    #if 'gray' in weights_file: continue
    #if 'grad' in weights_file: continue
    #if 'angle' in weights_file: continue
    #if 'height' in weights_file: continue
    #if 'all' in weights_file: continue
    if 'all' not in weights_file: continue
    '''
    if '5_' in weights_file: continue
    if 'heigth' in weights_file: continue
    #if '/home/mcube/weights_server_last/weights_type=semipyramid_3_08-23-2018_num=2000_gs_id=2_in=rgb_out=grad_epoch=100_NN=basic_aug=5' in weights_file: break
    print it_count, ' ', weights_file
    it_count += 1
    directory_save_test = weights_file[:-5] + '/'
    losses_train = np.load(directory_save_test + 'losses_train.npy')
    total_loss = 0
    std_total_loss = 0
    for i in range(4):
        total_loss += np.mean(losses_train[i])/4
        std_total_loss += np.std(losses_train[i])/4
    train_loss.append(total_loss)
    std_train_loss.append(std_total_loss)
    losses_test = np.load(directory_save_test + 'losses_test.npy')
    total_loss = 0
    std_total_loss = 0
    for i in range(4):
        total_loss += np.mean(losses_test[i])/4
        std_total_loss += np.mean(losses_test[i])/4
    test_loss.append(total_loss)
    std_test_loss.append(std_total_loss)
    max_heights = np.load(directory_save_test + 'max_heights.npy')
    heights.append(np.mean(max_heights))
    if os.path.isfile(directory_save_test + 'loss_thresholds_train.npy'):
        loss_thresholds_train = np.load(directory_save_test + 'loss_thresholds_train.npy')
        loss_thresholds_test = np.load(directory_save_test + 'loss_thresholds_test.npy')
        plt.plot(np.mean(loss_thresholds_train, axis=0))
        plt.plot(np.mean(loss_thresholds_test, axis=0))
        plt.show()
    
plt.errorbar(x =range(len(train_loss)), y = train_loss, yerr=std_train_loss, label='Train err')
#plt.errorbar(x =range(len(test_loss)), y = test_loss, yerr=std_test_loss, c='r', label='Test err')
plt.plot(range(len(test_loss)), test_loss, c='r', label='Test err')
#plt.show()
plt.plot(np.reciprocal(heights)*10-20, label='Inv height')
plt.legend()
plt.show()
