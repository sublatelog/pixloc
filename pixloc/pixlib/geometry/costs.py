import torch
from typing import Optional, Tuple
from torch import Tensor

from . import Pose, Camera
from .optimization import J_normalization
from .interpolation import Interpolator


class DirectAbsoluteCost:
    def __init__(self, 
                 interpolator: Interpolator, 
                 normalize: bool = False
                ):
        
        self.interpolator = interpolator
        self.normalize = normalize

    def residuals(
                    self, 
                    T_w2q: Pose, 
                    camera: Camera, 
                    p3D: Tensor,
                    F_ref: Tensor, 
                    F_query: Tensor,
                    confidences: Optional[Tuple[Tensor, Tensor]] = None,
                    do_gradients: bool = False
                ):

        """
        T_w2q
        torch.Size([1])
        Pose: torch.Size([1]) torch.float32 cuda:0
        
        p3D
        torch.Size([1, 512, 3])
        tensor([[[-9.5236, -8.5752, 18.2383],
                 [ 4.4923, -2.7068, 26.2318],
                 [-2.3577, -5.0674, 25.6186],
                 ...,
                 [ 7.5385, -6.5783, 24.9996],
                 [ 3.5002, -4.7065, 15.0571],
                 [ 2.7788,  1.8855, 14.9572]]], device='cuda:0')
                 
        p3D_q
        torch.Size([1, 512, 3])
        tensor([[[-9.7537, -8.4352, 17.7056],
                 [ 4.3295, -2.8587, 25.7892],
                 [-2.5611, -5.0860, 25.1334],
                 ...,
                 [ 7.3075, -6.7844, 24.5617],
                 [ 3.3583, -4.8078, 14.6038],
                 [ 2.7638,  1.7971, 14.5192]]], device='cuda:0',
               grad_fn=<AddBackward0>)
        """

        p3D_q = T_w2q * p3D

        # 3Dから2Dへ変換
        p2D, visible = camera.world2image(p3D_q)
        
        """
        p2D
        torch.Size([1, 512, 2])
        tensor([[[ 15.5255,  16.6838],
                 [143.4282,  75.8013],
                 [ 89.4386,  57.7025],
                 ...,
                 [166.7198,  45.2736],
                 [153.8433,  35.0925],
                 [147.7742, 122.5744]]], device='cuda:0', grad_fn=<AddBackward0>)
                 
        visible
        torch.Size([1, 512])
        tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                 False,  True,  True,  True,  True,  True,  True,  True,  True,  True,         
                 
        """
     
        
        F_p2D_raw, valid, gradients = self.interpolator(
                                                        F_query, 
                                                        p2D, 
                                                        return_gradients=do_gradients
                                                        )
        
        """
        F_p2D_raw
        torch.Size([1, 512, 128])
        tensor([[[ 0.0176,  0.0547, -0.0049,  ...,  0.1012, -0.0267, -0.0972],
                 [ 0.1280,  0.0754, -0.0377,  ..., -0.0520,  0.0066, -0.1085],
                 [ 0.0318,  0.0816, -0.0278,  ..., -0.0360, -0.0511, -0.1607],
                 ...,
                 [ 0.1011, -0.0117,  0.1192,  ...,  0.0488, -0.0466, -0.0822],
                 [ 0.0917,  0.0745, -0.0744,  ..., -0.0106, -0.0499, -0.0077],
                 [ 0.1989,  0.0932,  0.0100,  ..., -0.0280,  0.0017, -0.0286]]],
               device='cuda:0', grad_fn=<TransposeBackward0>)
        
        valid
        torch.Size([1, 512])
        tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                 False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        
        gradients
        torch.Size([1, 512, 128, 2])
        tensor([[[[ 0.0236, -0.0176],
                  [ 0.0041,  0.0024],
                  [ 0.1165,  0.0346],
                  ...,
                  [ 0.0401,  0.0305],
                  [-0.0183, -0.0301],
                  [-0.0115, -0.0069]],
        """
        
 
        valid = valid & visible

        if confidences is not None:
            C_ref, C_query = confidences
            C_query_p2D, _, _ = self.interpolator(
                                                    C_query, 
                                                    p2D, 
                                                    return_gradients=False
                                                    )
            
            weight = C_ref * C_query_p2D
            weight = weight.squeeze(-1).masked_fill(~valid, 0.)
            
        else:
            weight = None

        if self.normalize:
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_ref
        info = (p3D_q, F_p2D_raw, gradients)
        return res, valid, weight, F_p2D, info

    def jacobian(
                self, 
                T_w2q: Pose, 
                camera: Camera,
                p3D_q: Tensor, 
                F_p2D_raw: Tensor, 
                J_f_p2D: Tensor
                ):

        J_p3D_T = T_w2q.J_transform(p3D_q)
        J_p2D_p3D, _ = camera.J_world2image(p3D_q)

        if self.normalize:
            J_f_p2D = J_normalization(F_p2D_raw) @ J_f_p2D

        """
        
        
        J_p2D_p3D
        torch.Size([1, 512, 2, 3])
        tensor([[[[ 8.2646, -1.2098,  3.9765],
                  [-1.2057,  8.5969,  3.4315]],

                 [[ 7.6056,  0.1123, -1.2644],
                  [ 0.1119,  7.6725,  0.8317]],

                 [[ 7.8742, -0.1291,  0.7762],
                  [-0.1287,  7.6609,  1.5371]],

        J_p3D_T
        torch.Size([1, 512, 3, 6])
        tensor([[[[  1.0000,   0.0000,   0.0000,  -0.0000,  17.7056,   8.4352],
                  [  0.0000,   1.0000,   0.0000, -17.7056,  -0.0000,  -9.7537],
                  [  0.0000,   0.0000,   1.0000,  -8.4352,   9.7537,  -0.0000]],

                 [[  1.0000,   0.0000,   0.0000,  -0.0000,  25.7892,   2.8587],
                  [  0.0000,   1.0000,   0.0000, -25.7892,  -0.0000,   4.3295],
                  [  0.0000,   0.0000,   1.0000,  -2.8587,  -4.3295,  -0.0000]],

                 [[  1.0000,   0.0000,   0.0000,  -0.0000,  25.1334,   5.0860],
                  [  0.0000,   1.0000,   0.0000, -25.1334,  -0.0000,  -2.5611],
                  [  0.0000,   0.0000,   1.0000,  -5.0860,   2.5611,  -0.0000]],

        J_p2D_T
        torch.Size([1, 512, 2, 6])
        tensor([[[[ 8.2646e+00, -1.2098e+00,  3.9765e+00, -1.2122e+01,  1.8512e+02,
                    8.1513e+01],
                  [-1.2057e+00,  8.5969e+00,  3.4315e+00, -1.8116e+02,  1.2122e+01,
                   -9.4022e+01]],

                 [[ 7.6056e+00,  1.1232e-01, -1.2644e+00,  7.1789e-01,  2.0162e+02,
                    2.2228e+01],
                  [ 1.1194e-01,  7.6725e+00,  8.3170e-01, -2.0024e+02, -7.1401e-01,
                    3.3538e+01]],
        """
             
        J_p2D_T = J_p2D_p3D @ J_p3D_T
        
        J = J_f_p2D @ J_p2D_T
        
        return J, J_p2D_T

    def residual_jacobian(
                            self, 
                            T_w2q: Pose, 
                            camera: Camera, 
                            p3D: Tensor,
                            F_ref: Tensor, 
                            F_query: Tensor,
                            confidences: Optional[Tuple[Tensor, Tensor]] = None
                        ):

        res, valid, weight, F_p2D, info = self.residuals(
                                                        T_w2q, 
                                                        camera, 
                                                        p3D, 
                                                        F_ref, 
                                                        F_query, 
                                                        confidences, 
                                                        True
                                                        )
        J, _ = self.jacobian(T_w2q, camera, *info)
        
        return res, valid, weight, F_p2D, J
