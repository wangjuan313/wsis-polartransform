import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pylab as plt

def performance_summary(folder, dice_2d_all):
    epochs = list(dice_2d_all.keys())
    threshold = list(dice_2d_all[epochs[0]].keys())
    assert len(epochs)==50, len(epochs)
    mean_2d_array, std_2d_array = [], []
    for key in dice_2d_all.keys():
        mean_2d_array.append(np.mean(np.asarray(dice_2d_all[key]),axis=0))
        std_2d_array.append(np.std(np.asarray(dice_2d_all[key]),axis=0))
    mean_2d_array = np.vstack(mean_2d_array)
    std_2d_array = np.vstack(std_2d_array)
    
    max_mean = np.max(mean_2d_array)
    ind = np.where(mean_2d_array==np.max(mean_2d_array))
    max_std  = std_2d_array[ind][0]
    epoch = epochs[ind[0][0]]
    th = threshold[ind[1][0]]
    return max_mean, max_std, th, epoch
        
if __name__ == "__main__":
    dir_save_root = os.path.join('results','promise')
    weights_min = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    alphas = [0.5, 1, 2]
    folder1 = ['residual_parallel_approx_focal_40_20_expsumr=4_unary_pair',
               'residual_parallel_approx_focal_40_20_explogs=6_unary_pair',
               'residual_parallel_approx_focal_40_20_expsumr=4_unary_pair_margin=5',
               'residual_parallel_approx_focal_40_20_explogs=6_unary_pair_margin=5']    
    
    def get_polar_name(approx_method, alpha):
        if approx_method == 'softmax':
            folder = f'residual_polarw_approx_focal_90_30_expsumr={alpha}_unary_pair_margin=5'
        else:
            folder = f'residual_polarw_approx_focal_90_30_explogs={alpha}_unary_pair_margin=5'
        return folder
    
    def get_polar_assisting_name(approx_method, alpha, weight_min):
        if approx_method == 'softmax':
            folder = f'residual_parallel_polarw_{weight_min}_approx_focal_90_30_expsumr={alpha}_unary_pair_margin=5'
            if weight_min == 0.5:
                folder = f'residual_parallel_polarw_approx_focal_90_30_expsumr={alpha}_unary_pair_margin=5'
        else:
            folder = f'residual_parallel_polarw_{weight_min}_approx_focal_90_30_explogs={alpha}_unary_pair_margin=5'
            if weight_min == 0.5:
                folder = f'residual_parallel_polarw_approx_focal_90_30_explogs={alpha}_unary_pair_margin=5'
        return folder

    folder2 = [get_polar_name('softmax', alpha) for alpha in alphas]
    folder3 = [get_polar_name('quasimax', alpha) for alpha in alphas]
    folder4 = [get_polar_assisting_name('softmax', alpha, weight_min) for alpha in alphas for weight_min in weights_min]
    folder5 = [get_polar_assisting_name('quasimax', alpha, weight_min) for alpha in alphas for weight_min in weights_min]
    folders_list = [folder1, folder2, folder3, folder4, folder5]
    
    metrics = ['dice_3d']
    for metric in metrics:
        print('performance summary: {:s}'.format(metric).center(50,"#"))
        for k, folders in enumerate(folders_list):
            if k == 0:
                for folder in folders:
                    output_dir = os.path.join(dir_save_root, folder)
                    file_name = os.path.join(output_dir,metric+'.xlsx')
                    results = pd.read_excel(file_name, sheet_name=None)
                    max_mean, max_std, th, epoch = performance_summary(folder, results)
                    print('{:s}: {:.3f}({:.3f}), th={:0.3f}, epoch={:s}'.format(folder, max_mean, max_std, th, epoch))
            else:
                max_mean = 0
                for folder in folders:
                    output_dir = os.path.join(dir_save_root, folder)
                    file_name = os.path.join(output_dir,metric+'.xlsx')
                    results = pd.read_excel(file_name, sheet_name=None)
                    results = performance_summary(folder, results)
                    if max_mean < results[0]:
                        max_mean, max_std, th, epoch = results
                        experiment = folder
                print('{:s}: {:.3f}({:.3f}), th={:0.3f}, epoch={:s}'.format(experiment, max_mean, max_std, th, epoch))

    markers = ['-ro', ':+', '--s']
    for approx_method in ['softmax', 'quasimax']:
        approx_summary = []
        plt.figure()
        for marker, alpha in zip(markers, alphas):
            alpha_summary = []
            for weight_min in weights_min:
                folder = get_polar_assisting_name(approx_method, alpha, weight_min)
                output_dir = os.path.join(dir_save_root, folder)
                file_name = os.path.join(output_dir, 'dice_3d.xlsx')
                results = pd.read_excel(file_name, sheet_name=None)
                dice_mean, dice_std, _, _ = performance_summary(folder, results)
                dice_mean = np.around(dice_mean, 3)
                alpha_summary.append(dice_mean)
            approx_summary.append(alpha_summary)
            plt.plot(weights_min, alpha_summary, marker, markerfacecolor='none', linewidth=2, label=rf'$\alpha$={alpha}')
        plt.legend(loc=4)
        plt.ylim([0.84, 0.885])
        # plt.xlim([0, 1])
        plt.grid()
        plt.xlabel(r'$w_{min}$')
        plt.ylabel('Dice coefficient')
        plt.savefig(os.path.join(dir_save_root, f"{approx_method}.png"), dpi=300, bbox_inches='tight')
        
        