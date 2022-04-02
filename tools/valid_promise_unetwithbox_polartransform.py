import os, sys
import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torchvision_wj.datasets.samplers import PatientSampler
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unet_net import *
from torchvision_wj import pd_utils 
from torchvision_wj.utils.losses import *
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.promise_utils import get_promise
import torchvision_wj.utils.transforms as T

@torch.no_grad()
def evaluate(epoch, model, data_loader, image_names, device, threshold, save_detection=None, smooth=1e-10):
    file_2d = os.path.join(save_detection,'dice_2d.xlsx')
    file_3d = os.path.join(save_detection,'dice_3d.xlsx')
    torch.set_num_threads(1)
    model.eval()

    nn = 0
    dice_2d, dice_3d = {k:[] for k in range(len(threshold))}, {k:[] for k in range(len(threshold))}
    for images, targets in data_loader:
        nn = nn + 1
        # print("{}/{}".format(nn,len(data_loader))) 

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt = torch.stack([t["masks"] for t in targets], dim=0)
        gt = gt.bool()
        _, outputs = model(images, targets)
        
        for out in outputs:
            for n_th,th in enumerate(threshold):
                pred = out>th
                intersect = pred&gt
                v_dice_2d = (2*torch.sum(intersect,dim=(1,2,3))+smooth)/(torch.sum(pred,dim=(1,2,3))+torch.sum(gt,dim=(1,2,3))+smooth)
                v_dice_3d = (2*torch.sum(intersect)+smooth)/(torch.sum(pred)+torch.sum(gt)+smooth)
                dice_2d[n_th].append(v_dice_2d.cpu().numpy())
                dice_3d[n_th].append(v_dice_3d.cpu().numpy())

    dice_2d = [np.hstack(dice_2d[key]) for key in dice_2d.keys()]
    dice_3d = [np.hstack(dice_3d[key]) for key in dice_3d.keys()]
    dice_2d = np.vstack(dice_2d).T
    dice_3d = np.vstack(dice_3d).T
    
    dice_2d = pd.DataFrame(data=dice_2d, columns=threshold)
    dice_3d = pd.DataFrame(data=dice_3d, columns=threshold)
    
    pd_utils.append_df_to_excel(file_2d, dice_2d, sheet_name=str(epoch), index=False)
    pd_utils.append_df_to_excel(file_3d, dice_3d, sheet_name=str(epoch), index=False)

    mean_2d = np.mean(dice_2d, axis=0)
    std_2d = np.std(dice_2d, axis=0)
    loc2 = np.argmax(mean_2d)
    mean_3d = np.mean(dice_3d, axis=0)
    std_3d = np.std(dice_3d, axis=0)
    loc3 = np.argmax(mean_3d)
    print('2d mean: {}({})'.format(mean_2d[loc2],std_2d[loc2]))
    print('3d mean: {}({})'.format(mean_3d[loc3],std_3d[loc3]))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=1, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    dir_save_root = os.path.join('results','promise')
    threshold = [0.001,0.005,0.01]+list(np.arange(0.05,0.9,0.05))
    ## generalized mil
    if n_exp==1:
    	experiment_name = 'residual_parallel_approx_focal_40_20_expsumr=4_unary_pair_margin=5'
    elif n_exp==2:	
        experiment_name = 'residual_parallel_approx_focal_40_20_explogs=6_unary_pair_margin=5'
    ## polar transformation based mil
    elif n_exp==3:
        experiment_name = 'residual_polarw_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==4:
        experiment_name = 'residual_polarw_approx_focal_90_30_expsumr=1_unary_pair_margin=5'
    elif n_exp==5:
        experiment_name = 'residual_polarw_approx_focal_90_30_expsumr=2_unary_pair_margin=5'
    elif n_exp==6:
        experiment_name = 'residual_polarw_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==7:
        experiment_name = 'residual_polarw_approx_focal_90_30_explogs=1_unary_pair_margin=5'
    elif n_exp==8:
        experiment_name = 'residual_polarw_approx_focal_90_30_explogs=2_unary_pair_margin=5'
    ## polar transformation assisting mil (the proposed approach), alpha-sofmax
    elif n_exp==9:
        experiment_name = 'residual_parallel_polarw_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==10:
        experiment_name = 'residual_parallel_polarw_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==11:
        experiment_name = 'residual_parallel_polarw_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==12:
        experiment_name = 'residual_parallel_polarw_0.2_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==13:
        experiment_name = 'residual_parallel_polarw_0.2_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==14:
        experiment_name = 'residual_parallel_polarw_0.2_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==15:
        experiment_name = 'residual_parallel_polarw_0.3_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==16:
        experiment_name = 'residual_parallel_polarw_0.3_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==17:
        experiment_name = 'residual_parallel_polarw_0.3_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==18:
        experiment_name = 'residual_parallel_polarw_0.4_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==19:
        experiment_name = 'residual_parallel_polarw_0.4_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==20:
        experiment_name = 'residual_parallel_polarw_0.4_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==21:
        experiment_name = 'residual_parallel_polarw_0.6_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==22:
        experiment_name = 'residual_parallel_polarw_0.6_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==23:
        experiment_name = 'residual_parallel_polarw_0.6_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==24:
        experiment_name = 'residual_parallel_polarw_0.7_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==25:
        experiment_name = 'residual_parallel_polarw_0.7_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==26:
        experiment_name = 'residual_parallel_polarw_0.7_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==27:
        experiment_name = 'residual_parallel_polarw_0.8_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==28:
        experiment_name = 'residual_parallel_polarw_0.8_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    elif n_exp==29:
        experiment_name = 'residual_parallel_polarw_0.8_approx_focal_90_30_expsumr=0.5_unary_pair_margin=5'
    ## polar transformation assisting mil (the proposed approach), alpha-quasimax
    elif n_exp==30:
        experiment_name = 'residual_parallel_polarw_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==31:
        experiment_name = 'residual_parallel_polarw_approx_focal_90_30_explogs=1_unary_pair_margin=5'
    elif n_exp==32:
        experiment_name = 'residual_parallel_polarw_approx_focal_90_30_explogs=2_unary_pair_margin=5'
    elif n_exp==33:
        experiment_name = 'residual_parallel_polarw_0.2_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==34:
        experiment_name = 'residual_parallel_polarw_0.2_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==35:
        experiment_name = 'residual_parallel_polarw_0.2_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==36:
        experiment_name = 'residual_parallel_polarw_0.3_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==37:
        experiment_name = 'residual_parallel_polarw_0.3_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==38:
        experiment_name = 'residual_parallel_polarw_0.3_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==39:
        experiment_name = 'residual_parallel_polarw_0.4_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==40:
        experiment_name = 'residual_parallel_polarw_0.4_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==41:
        experiment_name = 'residual_parallel_polarw_0.4_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==42:
        experiment_name = 'residual_parallel_polarw_0.6_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==43:
        experiment_name = 'residual_parallel_polarw_0.6_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==44:
        experiment_name = 'residual_parallel_polarw_0.6_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==45:
        experiment_name = 'residual_parallel_polarw_0.7_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==46:
        experiment_name = 'residual_parallel_polarw_0.7_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==47:
        experiment_name = 'residual_parallel_polarw_0.7_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==48:
        experiment_name = 'residual_parallel_polarw_0.8_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==49:
        experiment_name = 'residual_parallel_polarw_0.8_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'
    elif n_exp==50:
        experiment_name = 'residual_parallel_polarw_0.8_approx_focal_90_30_explogs=0.5_unary_pair_margin=5'

    print(experiment_name)
    output_dir = os.path.join(dir_save_root, experiment_name)
    _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
    assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
    cfg = {'data_params': {'workers': 4}}
    _C = config.config_updates(_C, cfg)

    train_params       = _C['train_params']
    data_params        = _C['data_params']
    net_params         = _C['net_params']
    dataset_params     = _C['dataset']
    save_params        = _C['save_params']

    device = torch.device(_C['device'])

    def get_transform():
        transforms = []
        transforms.append(T.ToTensor())
        transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
        return T.Compose(transforms)

    # Data loading code
    print("Loading data")
    dataset_test = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['valid_path'][0], 
                                   gt_folder=dataset_params['valid_path'][1], 
                                   transforms=get_transform(),
                                   transform_generator=None, visual_effect_generator=None)
    image_names = dataset_test.image_names
    
    print("Creating data loaders")
    test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

    data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

    print("Creating model with parameters: {}".format(net_params))
    
    net = eval(net_params['model_name'])(net_params['input_dim'], net_params['seg_num_classes'],
                                             net_params['softmax'])
    losses, loss_weights = [], []
    for loss in net_params['losses']:
        losses.append(eval(loss[0])(**loss[1]))
        loss_weights.append(loss[2])
    model = UNetWithBox(net, losses, loss_weights, softmax=net_params['softmax'])
    model.to(device)
    
    file_2d = os.path.join(output_dir,'dice_2d.xlsx')
    file_3d = os.path.join(output_dir,'dice_3d.xlsx')
    if os.path.exists(file_2d):
        os.remove(file_2d)
    if os.path.exists(file_3d):
        os.remove(file_3d)
    for epoch in range(50):
        model_file = 'model_{:02d}'.format(epoch)
        print('loading model {}.pth'.format(model_file))
        checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
        print('start evaluating {} ...'.format(epoch))
        model.training = False
        evaluate(epoch, model, data_loader_test, image_names=image_names, device=device, threshold=threshold, save_detection=output_dir)
        
    
    dice_2d_all = pd.read_excel(file_2d, sheet_name=None)
    dice_3d_all = pd.read_excel(file_3d, sheet_name=None)



    