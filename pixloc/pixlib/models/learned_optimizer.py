import logging
from typing import Tuple, Optional
import torch
from torch import nn, Tensor

from .base_optimizer import BaseOptimizer
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry import losses  # noqa

logger = logging.getLogger(__name__)


class DampingNet(nn.Module):
    def __init__(self, conf, num_params=6):
        super().__init__()
        self.conf = conf
        
        if conf.type == 'constant':
            const = torch.zeros(num_params)
            self.register_parameter('const', torch.nn.Parameter(const))
        else:
            raise ValueError(f'Unsupported type of damping: {conf.type}.')

    def forward(self):
        min_, max_ = self.conf.log_range
        lambda_ = 10.**(min_ + self.const.sigmoid()*(max_ - min_))
        return lambda_


class LearnedOptimizer(BaseOptimizer):
    default_conf = dict(
                        damping=dict(
                                    type='constant',
                                    log_range=[-6, 5],
                                    ),
                        feature_dim=None,

                        # deprecated entries
                        lambda_=0.,
                        learned_damping=True,
                        )

    def _init(self, conf):
        self.dampingnet = DampingNet(conf.damping)
        assert conf.learned_damping
        super()._init(conf)

    def _run(self, 
             p3D: Tensor, 
             F_ref: Tensor, 
             F_query: Tensor,
             T_init: Pose,
             camera: Camera, 
             mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None
            ):

        """
        p3D
        tensor([[[ -3.4736,  -1.1382,  17.3129],
                 [-18.7738,  -6.7321,  39.3825],
                 [ -1.7312,   1.9992,   9.2566],
                 ...,
                 [  0.8362,  -7.9434,  19.0898],
                 [-19.1580,  -7.2810,  38.8155],
                 [-13.0297,   0.9339,  28.3888]]], device='cuda:0')
                 
        T_init
        Pose: torch.Size([1]) torch.float32 cuda:0
        
        camera
        Camera torch.Size([1]) torch.float32 cuda:0
        
        W_ref_query
        (tensor([[[0.6974],
                 [0.7410],
                 [0.7573],
                 [0.5812],
                 [0.6925],
                 [0.7745],
                 [0.7493],
                 [0.6648],
                 [0.6243],
                 [0.6669],
                 
        tensor([[[[0.7303, 0.6751, 0.7109,  ..., 0.6711, 0.8203, 0.7823],
                  [0.7899, 0.7414, 0.7441,  ..., 0.6336, 0.8918, 0.6993],
                  [0.7736, 0.8314, 0.6970,  ..., 0.7183, 0.7811, 0.7448],
                  ...,
                  [0.6738, 0.6486, 0.6448,  ..., 0.6467, 0.6656, 0.6527],
                  
                  
        """
        print("p3D")
        print(p3D.shape)
        print("T_init")
        print(T_init.shape)
        print("camera")
        print(camera.shape)
        print("W_ref_query")
        print(W_ref_query[0].shape)
        print(W_ref_query[1].shape)
        
        
        T = T_init
        J_scaling = None
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
            
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()

        for i in range(self.conf.num_iters):
            res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(T, *args)
            
            """
            res
            tensor([[[-0.1436, -0.1194,  0.0304,  ..., -0.1115, -0.0853,  0.0285],
                     [-0.0686,  0.0023,  0.0738,  ...,  0.0321,  0.0378, -0.0951],
                     [ 0.0402,  0.0589,  0.0306,  ..., -0.0885,  0.1199,  0.0130],
                     ...,
                     [ 0.0929,  0.0983, -0.0624,  ..., -0.0343, -0.0415,  0.0408],
                     [ 0.0073, -0.0117, -0.0397,  ..., -0.0173, -0.0596, -0.0430],
                     [ 0.1085, -0.0488,  0.1766,  ...,  0.0086, -0.0165, -0.0436]]],
                   device='cuda:0')
                   
            valid
            tensor([[ True,  True,  True,  True,  True,  True, False,  True,  True,  True,
                      True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                      True,  True,  True, False,  True,  True,  True,  True,  True,  True,
                      True,  True,  True,  True,  True,  True,  True,  True, False,  True,
                      True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                      True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                     False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                     
            w_unc
            tensor([[0.4777, 0.5401, 0.4450, 0.3916, 0.5184, 0.5744, 0.0000, 0.4875, 0.5237,
                     0.5372, 0.4048, 0.3982, 0.5854, 0.4604, 0.4574, 0.5753, 0.4767, 0.4203,
                     0.3708, 0.4456, 0.3526, 0.4879, 0.3846, 0.0000, 0.4192, 0.4770, 0.3175,
                     0.4678, 0.4329, 0.3454, 0.3915, 0.5181, 0.6028, 0.4879, 0.4264, 0.3582,
                     0.6041, 0.4329, 0.0000, 0.4112, 0.5046, 0.3950, 0.4686, 0.4519, 0.3852,
                     0.5015, 0.5190, 0.5201, 0.6604, 0.4813, 0.3981, 0.5150, 0.3686, 0.3549,
            
            J
            tensor([[[[ 2.0140e-01,  9.0354e-01,  1.0955e-01, -1.5505e+01,  3.8853e+00,
                       -3.5399e+00],
                      [ 3.0879e-01,  3.0742e-01,  9.6141e-02, -5.3422e+00,  5.6573e+00,
                       -9.3190e-01],
                      [ 1.5647e-01,  4.7464e-02,  4.1500e-02, -8.5504e-01,  2.8366e+00,
                       -2.0468e-02],
                      ...,
                      [-2.6803e-01,  6.3152e-01, -2.3586e-02, -1.0723e+01, -4.6609e+00,
                       -2.9379e+00],
                      [ 1.2500e-01, -6.1058e-01, -1.0062e-02,  1.0405e+01,  2.0858e+00,
                        2.6883e+00],
                      [ 1.5818e-01,  4.0377e-01,  6.5661e-02, -6.9477e+00,  2.9664e+00,
                       -1.5046e+00]],
            """
            
            print("res")
            print(res.shape)
            print("valid")
            print(valid.shape)
            print("w_unc")
            print(w_unc.shape)
            print("J")
            print(J.shape)
            
            if mask is not None:
                valid &= mask
                
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            # compute the cost and aggregate the weights
            cost = (res**2).sum(-1)
            cost, w_loss, _ = self.loss_fn(cost)
            weights = w_loss * valid.float()
            
            if w_unc is not None:
                weights *= w_unc
                
            if self.conf.jacobi_scaling:
                J, J_scaling = self.J_scaling(J, J_scaling, valid)

            # solve the linear system
            g, H = self.build_system(J, res, weights)
            
            delta = optimizer_step(g, H, lambda_, mask=~failed)
            
            if self.conf.jacobi_scaling:
                delta = delta * J_scaling

            # compute the pose update
            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost, valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
            
            if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost):
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed
