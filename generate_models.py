import sys
#sys.path.append('.')

from experiment import PruningExperiment, PruningClass, TrainingExperiment
from csv_analysis import *
import os
import numpy as np

def _makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ROOT='./'
# Replace with absolute or relative paths to shrinkbench and trainning data
data_root = './Training_data'
os.environ['DATAPATH'] = data_root
_makedir(os.path.join(data_root, 'CIFAR100'))
_makedir(os.path.join(data_root, 'CIFAR10'))
_makedir('./saved_models')
_makedir('./results')
_makedir('./logs')
_makedir('./saved_models')
os.environ["ShrinkPATH"] = './'
models = ['resnet20', 'resnet32', 'resnet44', 'resnet110', 'resnet56']


def run_cifar10(model_arch, rounds=[]):
    # This code will rely on the existence of a folder called 'saved_models' in the shrinkbench directory, you 
    # will probably need to create it such that `shrinkbench/save_models` exists if it currently does not
    	# 30 models on the 4 different resnet models, mixed pruning
    
    
    assert(model_arch in models, f"{model_arch} not {models}")
    
    
    exp = PruningClass(dataset='CIFAR10', 
                        model=f'{model_arch}',  
                        train_kwargs={
                            'optim': 'SGD',
                            'epochs': 10,
                            'lr': 1e-2,
                            'weight_decay' : 5e-4},
                        dl_kwargs={'batch_size':128},
                      save_freq=1)
    
    exp.run_init()
    strategies = ["MixedMagGrad"] 
    compressions = [2, 4, 10, 20, 50]
    for strategy in strategies:
        exp.fix_seed()
        for i in rounds:
            exp.round = i
    #         Change ResultPATH to the desired folder (should be different folder for each strategy)
    #         I find it easier to just make an empty folder with same name as model arch to tell them apart
            os.environ["ResultPATH"] = f'{ROOT}/results/{model_arch}'
    
            exp.state = 'Original'
            exp.compression = 0
            exp.pruning = False
            exp.build_model(f"{model_arch}")
            exp.update_optim(epochs=10, lr=1e-2)
            
            exp.run()
            exp.update_optim(epochs=20, lr=1e-1)
            exp.run()
            for x in [1e-2, 1e-3, 1e-4]:
                exp.update_optim(epochs=10, lr=x)
                exp.run()
    
            cp = exp.load_model(checkpoint=True)
            exp.build_model(f"{model_arch}")
            exp.to_device()
            exp.model.load_state_dict(cp['model_state_dict'])
            exp.optim.load_state_dict(cp['optim_state_dict'])
            exp.eval()
    
            exp.save_model(f"{i}-{model_arch}-{exp.compression}")
                
            cifar10_init(exp, i)
            cifar10_log(exp, i)
            
            exp.strategy = strategy
            for compression in compressions:
                exp.compression = compression
                exp.prune()
                exp.state = "Compressed"
                cifar10_log(exp, i)
    
                exp.state = "Finetuned"
                exp.update_optim(epochs=10, lr=1e-1)
                exp.run()
                for x in [1e-2, 1e-3, 1e-4]:
                    exp.update_optim(epochs=5, lr=x)
                    exp.run()
                    
                cp = exp.load_model(prune=True)
                exp.build_model(f"{model_arch}")
                exp.to_device()
                exp.prune()
                exp.model.load_state_dict(cp['model_state_dict'])
                exp.optim.load_state_dict(cp['optim_state_dict'])
                exp.eval()
                exp.update_optim('SGD', 15, 1e-2)
                exp.state = "FineTuned"
                exp.save_model(f"{i}-{model_arch}-{exp.compression}")
    
                cifar10_log(exp, i)
    
                checkpoint = exp.load_model(f"{i}-{model_arch}-0")
                exp.build_model(f"{model_arch}")
                exp.to_device()
                exp.model.load_state_dict(checkpoint['model_state_dict'])
                exp.optim.load_state_dict(checkpoint['optim_state_dict'])
                exp.eval()
                exp.update_optim(epochs=10, lr=1e-1)


def run_cifar100(rounds=[]):
    # 20 models on the CIFAR100 / resnet 56
    
    exp = PruningClass(dataset='CIFAR100', 
                        model='resnet56_C',  
                        train_kwargs={
                            'optim': 'SGD',
                            'epochs': 10,
                            'lr': 1e-2,
                            'weight_decay' : 5e-4},
                        dl_kwargs={'batch_size':128},
                      save_freq=1)
    
    exp.run_init()
    strategies = ["MixedMagGrad"] 
    compressions = [2, 4, 10, 20, 50]
    model_arch = 'resnet56_C'
    for strategy in strategies:
        exp.fix_seed()
        #for i in range(10, 30):
        for i in rounds:
            exp.round = i
    #         Change ResultPATH to the desired folder (should be different folder for each strategy)
            os.environ["ResultPATH"] = f'{ROOT}/results/{model_arch}'
            #os.environ["ResultPATH"] = f'/uusoc/exports/scratch/xiny/project/shrinkbench/Aidan/shrinkbench/'
    
            exp.state = 'Original'
            exp.compression = 0
            exp.pruning = False
            exp.build_model("resnet56_C")
            exp.update_optim(epochs=10, lr=1e-2)
            
            exp.run()
            exp.update_optim(epochs=20, lr=1e-1)
            exp.run()
            for x in [1e-2, 1e-3, 1e-4]:
                exp.update_optim(epochs=10, lr=x)
                exp.run()
    
            cp = exp.load_model(checkpoint=True)
            exp.build_model("resnet56_C")
            exp.to_device()
            exp.model.load_state_dict(cp['model_state_dict'])
            exp.optim.load_state_dict(cp['optim_state_dict'])
            exp.eval()
    
            exp.save_model(f"{i}-resnet56_C-{exp.compression}")
                
            cifar100_init(exp, i)
            cifar100_class(exp, i)
            cifar100_log(exp, i)
            
            exp.strategy = strategy
            for compression in compressions:
                exp.compression = compression
                exp.prune()
                exp.state = "Compressed"
                cifar100_class(exp, i)
                cifar100_log(exp, i)
    
                exp.state = "Finetuned"
                exp.update_optim(epochs=10, lr=1e-1)
                exp.run()
                for x in [1e-2, 1e-3, 1e-4]:
                    exp.update_optim(epochs=5, lr=x)
                    exp.run()
                    
                cp = exp.load_model(prune=True)
                exp.build_model("resnet56_C")
                exp.to_device()
                exp.prune()
                exp.model.load_state_dict(cp['model_state_dict'])
                exp.optim.load_state_dict(cp['optim_state_dict'])
                exp.eval()
                exp.update_optim('SGD', 15, 1e-2)
                exp.state = "FineTuned"
                exp.save_model(f"{i}-resnet56_C-{exp.compression}")
    
    
                cifar100_class(exp, i)
                cifar100_log(exp, i)
    
                checkpoint = exp.load_model(f"{i}-resnet56_C-0")
                exp.build_model("resnet56_C")
                exp.to_device()
                exp.model.load_state_dict(checkpoint['model_state_dict'])
                exp.optim.load_state_dict(checkpoint['optim_state_dict'])
                exp.eval()
                exp.update_optim(epochs=10, lr=1e-1)


if __name__ == '__main__':
    # for ciar10,  resnet20/resnet32/resnet44/resnet110, 0/1/2/3/4/5
    # for cifar100,resnet56 0/1/2/3
    dataset = sys.argv[1]
    model_arch = sys.argv[2] 
    idx   = int(sys.argv[3]) # 0,1,2,3,4,5
    if len(sys.argv) > 4:
        gpu = sys.argv[4]
    else:
        gpu = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    models_opt = ['resnet20', 'resnet32', 'resnet44', 'resnet110', 'resnet56']
    idx_opt = [0,1,2,3,4,5]
    assert(model_arch in models_opt)
    assert(idx in idx_opt)
    #one job runs 5 rounds for each model 
    N = 5
    rounds = np.arange(idx*N, (idx+1)*N)
    if dataset == "cifar10":
        run_cifar10(model_arch, rounds)	
    elif dataset == 'cifar100':
        run_cifar100(rounds+10)	
    else:
        print('No such dataset')
