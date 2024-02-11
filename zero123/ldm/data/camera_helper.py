"""Camera class.
"""
import math
import torch
import numpy as np
from PIL import Image
from typing import NamedTuple, Union, Optional
from kaolin.render.camera import Camera as KCamera
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

######################################
##### Gaussian Splatting Cameras #####
######################################


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: np.array
    image_path: str
    depth_path: str
    mask_path: str
    width: int
    height: int


class GSCamera:
    """Camera class for Gaussian Splatting (compatibility haven't been merged)."""

    def __init__(self,
                 world_view_transform, # world to camera
                 full_proj_transform, # world to screen
                 image_path=None,
                 image_width=512,
                 image_height=512,
                 mask_path=None,
                 depth_path=None,
                 fov_x=0.2443,
                 fov_y=0.2443,
                 znear=0.01,
                 zfar=100,
                 kao_cam=None):
        self.image = None
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.mask_path = mask_path
        self.depth_path = depth_path
        self.FoVx = fov_x
        self.FoVy = fov_y
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.kao_cam = kao_cam
        if world_view_transform is not None:
            view_inv = torch.inverse(world_view_transform)
            self.camera_center = view_inv[3][:3]

    def __repr__(self) -> str:
        viz_fn = lambda x : '[' + ', '.join([f'{t:.3f}' for t in x]) + ']' \
            if isinstance(x, torch.Tensor) and x.ndim == 1 else f'{x:.3f}'

        #viz_fn = lambda x : x
        return f'GSCamera(FoVx={viz_fn(self.FoVx)} FoVy={viz_fn(self.FoVy)} world2cam={self.world_view_transform.shape} proj={self.full_proj_transform.shape} device={self.world_view_transform.device} image={self.image_path} depth={self.depth_path} mask={self.mask_path}\n'

    def to(self, device):
        """Move to device."""
        if self.image is not None:
            self.image = self.image.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

    def as_dict(self):
        """Return a dictionary of attributes."""
        attrs = ['image_path', 'depth_path', 'mask_path',
                 'image', 'image_height', 'image_width',
                 'FoVx', 'FoVy', 'camera_center',
                 'world_view_transform', 'full_proj_transform']
        return {f'_camdata_{k}': getattr(self, k) for k in attrs \
                if getattr(self, k) is not None}

    @property
    def quat(self):
        cam2world = torch.linalg.inv(self.world_view_transform.T)
        return matrix_to_quaternion(cam2world[:3, :3])

    @staticmethod
    def from_dict(dic):
        """Load the attributes from a dictionary. Support a batch of cameras."""
        idx = len('_camdata_')
        if not isinstance(dic['_camdata_FoVx'], float):
            n_cams = len(dic['_camdata_FoVx'])
            cams = []
            for i in range(n_cams):
                cam = GSCamera(None, None)
                for k, v in dic.items():
                    setattr(cam, k[idx:], v[i])
                cams.append(cam)
        else:
            cam = GSCamera(None, None)
            for k, v in dic.items():
                setattr(cam, k[idx:], v)
            cams = cam
        return cams

    def load_image(self):
        """Deprecated. Load the image only when requested."""
        if self.image is not None:
            return self.image
        image = np.array(Image.open(self.image_path))
        image = torch.from_numpy(image) / 255.0
        self.image = image.permute(2, 0, 1)
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]
        return self.image

    def load_depth(self):
        """Load the depth only when requested."""
        if self.depth_path is None:
            return None
        if self.depth is not None:
            return self.depth
        depth = np.array(Image.open(self.depth_path))
        self.depth = torch.from_numpy(depth).float()
        return self.depth

    def load_mask(self):
        """Load the mask only when requested."""
        if self.mask_path is None:
            return None
        if self.mask is not None:
            return self.mask
        mask = np.array(Image.open(self.mask_path))
        self.mask = torch.from_numpy(mask).float()
        return self.mask

    @staticmethod
    def from_compact(c: torch.Tensor, **kwargs):
        """
        Args:
            c: [N, 25]. Extrinsics + Intrinsics.
        """
        cam2world_colmap = c[:, :16].reshape(-1, 4, 4)
        world2cam_colmap = torch.linalg.inv(cam2world_colmap)
        intrinsics = c[:, 16:].reshape(-1, 3, 3)
        fov_xs = 2 * torch.atan(1 / intrinsics[:, 0, 0])
        fov_ys = 2 * torch.atan(1 / intrinsics[:, 1, 1])
        #print(world2cam_colmap @ torch.Tensor([0, 0, 0, 1]).to(world2cam_colmap))
        return [GSCamera.from_matrix(w2c, float(fov_x), float(fov_y), **kwargs)
                for w2c, fov_x, fov_y in \
                zip(world2cam_colmap, fov_xs, fov_ys)]

    def to_compact(self):
        """
        Args:
            c: [N, 25]. Extrinsics + Intrinsics.
        """
        cam2world_colmap = torch.linalg.inv(self.world_view_transform.T)
        intrinsics = intrinsics_from_fov(self.FoVx, self.FoVy).to(cam2world_colmap)
        return torch.cat([cam2world_colmap.reshape(-1), intrinsics.view(-1)])
    
    @staticmethod
    def to_pixelsplat(cameras, device=None):
        """
        Returns:
            {
                image: [N, 1, C, H, W], batch view channel H W
                extrinsics: [N, 1, 4, 4],
                intrinsics: [N, 1, 3, 3],
                near: [N, 1],
                far: [N, 1],
            }
        """
        dic = {}
        device = device if device is not None else \
            cameras[0].world_view_transform.device
        #ones = torch.ones((len(cameras), 1)).to(device)
        intrinsics = lambda x: torch.Tensor([
            [0.5 / math.tan(x/2), 0, 0.5],
            [0, 0.5 / math.tan(x / 2), 0.5],
            [0, 0, 1]])
        dic['image'] = None if cameras[0].image is None else \
            torch.stack([c.image for c in cameras])[:, None].to(device)
        dic['extrinsics'] = torch.stack([
            torch.linalg.inv(c.world_view_transform.T)
            for c in cameras])[:, None].to(device)
        dic['intrinsics'] = torch.stack([
            intrinsics(c.FoVx) for c in cameras])[:, None].to(device)
        #dic['depth_near'] = ones * 1e-2
        #dic['depth_far'] = ones * 1e2
        return dic

    @staticmethod
    def from_info(cam_info: CameraInfo):
        """Creates a Camera object from a CameraInfo."""
        znear = 0.01
        zfar = 100
        world2view = world2view_from_rt(cam_info.R, cam_info.T)
        world_view_transform = torch.Tensor(world2view).transpose(0, 1)
        K = perspective_matrix_colmap(
            znear=znear, zfar=zfar,
            fov_x=cam_info.FovX, fov_y=cam_info.FovY)
        full_proj_transform = world_view_transform[None].bmm(K.T[None])[0]
        return GSCamera(world_view_transform, full_proj_transform,
                        fov_x=cam_info.FovX, fov_y=cam_info.FovY,
                        image_path=cam_info.image_path,
                        depth_path=cam_info.depth_path,
                        mask_path=cam_info.mask_path)

    @staticmethod
    def from_matrix(world_view_transform, fov_x, fov_y, **kwargs):
        """Creates a Camera object from a CameraInfo.
        Args:
            FoV: Field of view in radian.
        """
        znear = 0.01
        zfar = 100
        proj_matrix = perspective_matrix_colmap(
            znear=znear, zfar=zfar,
            fov_x=fov_x, fov_y=fov_y).T.to(world_view_transform)
        full_proj_transform = world_view_transform.T @ proj_matrix
        return GSCamera(world_view_transform.T, full_proj_transform,
                        fov_x=fov_x, fov_y=fov_y, **kwargs)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Taken from PyTorch3D.
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Taken from PyTorch3D.
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def compact2camera_quat(c):
    """Convert compact camera representation to camera matrix.
    Args:
        c: [N, 6]. quat, fov, cam_dist
    Returns:
        cam2world: [N, 4, 4]. camera to world matrix
        intrinsics: [N, 3, 3]. camera intrinsics
    """
    quats = c[:, :4]
    fov = c[:, 4]
    cam_dist = c[:, 5]

    Rs = [Rotation.from_quat(q).as_matrix() for q in quats]

    t = np.eye(4, dtype=np.float32)
    t[2, 3] = -cam_dist

    # convert to OpenCV camera
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    # world2cam -> cam2world
    cam2world = [np.linalg.inv(convert @ t @ R) for R in Rs]
    cam2world = np.stack(cam2world, axis=0)

    intrinsics = np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0]])[None].repeat(c.shape[0], 1, 1)
    intrinsics[:, 0, 0] = torch.tan(fov / 2)
    intrinsics[:, 1, 1] = torch.tan(fov / 2)
    
    return cam2world, intrinsics


def world2view_from_rt(R, t):
    """Get world to view matrix from rotation and translation.
    Args:
        R: torch.Tensor, [3, 3], rotation matrix.
        t: torch.Tensor, [3, ], translation vector.
    """
    if isinstance(R, np.ndarray):
        Rt = np.zeros((4, 4))
    else:
        Rt = torch.zeros((4, 4))

    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    if isinstance(R, np.ndarray):
        return Rt.astype('float32')
    return Rt.float()


def intrinsics_from_fov(
        fov_x: float,
        fov_y: float):
    """Get the camera intrinsics matrix from FoV.
    Notice that this transforms points into screen space [0, 1]^2 rather than NDC.
    .----> x
    |
    |
    v y
    Args:
        fov_x: Field of View in x-axis (degrees).
        fov_y: Field of View in y-axis (degrees).
        znear: near plane.
        zfar: far plane.
    """
    fx = 1 / math.tan(fov_y / 180 * math.pi / 2)
    fy = 1 / math.tan(fov_x / 180 * math.pi / 2)

    return torch.Tensor([
        [fx, 0, 0.5],
        [0, -fy, 0.5],
        [0, 0, 1]
    ])


def perspective_matrix_colmap(znear, zfar, fov_x, fov_y):
    """Get the colmap camera perspective matrix from FoV."""
    tanHalfFovY = math.tan(fov_y / 2)
    tanHalfFovX = math.tan(fov_x / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def intrinsics_to_fov(intrinsics):
    """Convert intrinsics to FoV.
    Args:
        intrinsics: a tensor of shape [batch, 3, 3]
    Returns:
        FoVx, FoVy: a tensor of shape [batch], in radians.
    """
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    ones = torch.ones((fx.shape[0],)).to(fx)
    return 2 * torch.atan2(ones, fx), 2 * torch.atan2(ones, fy)



#############################
##### EG3D Camera Utils #####
#############################


def angle_from_world2cam(world2cam, is_colmap=True):
    """Convert world2cam matrix to euler angles.
    Returns:
        Euler angle in radian.
    """
    if is_colmap:
        cam2world = torch.linalg.inv(world2cam)
        cam2world[..., 1] *= -1 # convert back to kaolin coordinate
        cam2world[..., 2] *= -1
        world2cam = torch.linalg.inv(cam2world)
    Rs = world2cam[:, :3, :3]
    return torch.from_numpy(Rotation.from_matrix( # angles: (N, 3), [yaw, pitch, roll]
        Rs.cpu().numpy()).as_euler('yxz')).float()


def perturb_camera(cam: GSCamera, da=0.1):
    radius = cam.camera_center.norm()
    angles = angle_from_world2cam(
        cam.world_view_transform.T[None], is_colmap=True)
    dh, dv, dr = sample_delta_angle(da, da, da, angles.shape[0])
    new_angles = angles + torch.stack([dh, dv, dr], 1)
    new_cam = make_colmap_camera(new_angles, radius, cam.FoVx)[0]
    new_cam.image_height = cam.image_height
    new_cam.image_width = cam.image_width
    return new_cam


def pos_from_angle(
        azim: torch.Tensor,
        elev: torch.Tensor,
        radius: torch.Tensor):
    """Create point from angles and radius.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians. 0 is z-axis.
        elev: polar angle (angle from the y axis) in radians. 0 is z-axis.
        radius: distance.
    """
    cos_elev = torch.cos(elev)
    x = cos_elev * torch.sin(azim)
    z = cos_elev * torch.cos(azim)
    y = torch.sin(elev)
    return radius * torch.stack([x, y, z], -1)


def sample_delta_angle(
        azim_std: float = 0.,
        elev_std: float = 0.,
        roll_std: float = 0.,
        n_sample: int = 1,
        device: str = 'cpu'):
    """
    Sample a delta angle from a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        n_sample: number of samples to return
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """
    dh = torch.rand((n_sample,), device=device) * azim_std
    dv = torch.randn((n_sample,), device=device) * elev_std
    dr = torch.randn((n_sample,), device=device) * roll_std
    return dh, dv, dr


def angle2matrix(angles):
    """get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [N, 3], yaw, pitch, roll
    Returns:
        R: [3, 3]. rotation matrix.
    """
    cos_y, cos_x, cos_z = torch.cos(angles).T
    sin_y, sin_x, sin_z = torch.sin(angles).T
    R = torch.eye(3)[None].repeat(angles.shape[0], 1, 1).to(angles)

    Rx = R.clone() # rotate around x: pitch
    Rx[:, 1, 1] = cos_x
    Rx[:, 1, 2] = -sin_x
    Rx[:, 2, 1] = sin_x
    Rx[:, 2, 2] = cos_x

    Ry = R.clone() # rotate around y: yaw
    Ry[:, 0, 0] = cos_y
    Ry[:, 0, 2] = sin_y
    Ry[:, 2, 0] = -sin_y
    Ry[:, 2, 2] = cos_y

    Rz = R.clone() # rotate around z: roll
    Rz[:, 0, 0] = cos_z
    Rz[:, 0, 1] = -sin_z
    Rz[:, 1, 0] = sin_z
    Rz[:, 1, 1] = cos_z
    # yaw -> pitch -> roll
    return torch.bmm(Rz, torch.bmm(Rx, Ry))


def make_colmap_camera(angles, radius, fov):
    """
    Args:
        angles: (N, 3), [yaw, pitch, roll]
    """
    R = angle2matrix(angles)
    # build world2cam matrix
    world2cams = torch.eye(4)[None].repeat(angles.shape[0], 1, 1)
    world2cams[:, :3, :3] = R
    world2cams[:, 2, 3] = -radius
    cam2worlds = torch.linalg.inv(world2cams)
    cam2worlds[..., 1] *= -1
    cam2worlds[..., 2] *= -1
    world2cams = torch.linalg.inv(cam2worlds)
    return [GSCamera.from_matrix(w2c, fov, fov) for w2c in world2cams]


def sample_lookat_camera(
        azim: float,
        elev: float,
        look_at: torch.Tensor = torch.zeros((3,)),
        azim_std: float = 0.,
        elev_std: float = 0.,
        fov: float = 0.2443,
        radius: float = 2.7,
        n_sample: int = 1,
        resolution: int = 512,
        device: str = 'cpu'):
    """Deprecated
    Sample a camera pose looking at a point with a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        look_at: 3-vector, point to look at
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        fov: field of view in radians
        radius: distance from camera to look_at point
        n_sample: number of samples to return
        resolution: image resolution of the camera
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """

    dh, dv = sample_delta_angle(azim_std, elev_std, 0, n_sample, device)[:2]
    h = dh + azim
    v = dv + elev
    cam_origs = pos_from_angle(h, v, radius)
    common_kwargs = {
        'at': look_at.to(device),
        'up': torch.Tensor([0., 1., 0.]).to(device),
        'fov': fov,
        'width': resolution,
        'height': resolution,
        'device': device
    }
    #print(f'sampling camera: at {common_kwargs["at"].device}, {cam_origs.device}')
    cams = [kaolin2colmap_cam(KCamera.from_args(
            eye=x, **common_kwargs)) for x in cam_origs]
    return cams


def R2compact(R_in, cam_dist=2.7, FoV=0.2443):
    '''
    Input a rotation matrix, output 25 dim label matrix (16 dim extrinsic + 9 dim intrinsic)
    Args:
        R_in: [3, 3] rotation matrix, rotating world coordinate to camera coordinate.
    '''
    n = R_in.shape[0]
    f = 1 / math.tan(FoV / 2)
    intrinsics = np.array([[f, 0, 0.5], [0, f, 0.5], [0, 0, 1]],
                          dtype=np.float32)
    intrinsics = np.repeat(intrinsics[None], n, axis=0)

    # build world2cam matrix
    world2cam = np.eye(4, dtype=np.float32)[None]
    world2cams = np.repeat(world2cam, n, axis=0)
    world2cams[:, :3, :3] = R_in
    world2cams[:, 2, 3] = - cam_dist

    cam2worlds = np.linalg.inv(world2cams)
    # convert to OpenCV camera
    cam2worlds[..., 1] *= -1
    cam2worlds[..., 2] *= -1

    # add intrinsics
    label_new = np.concatenate([
        cam2worlds.reshape(-1, 16),
        intrinsics.reshape(-1, 9)], -1)
    return label_new


def angle2compact(angles, cam_dist=2.7, FoV=0.2443):
    """Convert a rotation matrix to a compact representation."""
    R = angle2matrix(angles)
    return R2compact(R, cam_dist, FoV)


def kaolin2colmap_cam2world(kaolin_cam2world: torch.Tensor):
    """Convert a Kaolin camera to world matrix to a Colmap matrix.
    COLMAP (OpenCV)'s cooredinate system uses -y and -z than Kaolin.
    """
    sign = torch.Tensor([1, -1, -1, 1]).expand_as(kaolin_cam2world)
    return kaolin_cam2world * sign.to(kaolin_cam2world)


def kaolin2colmap_cam(kaolin_cam: KCamera):
    """Convert a Kaolin camera to a Colmap camera."""
    fov_x = float(kaolin_cam.intrinsics.fov_x) / 180 * math.pi # degree
    fov_y = float(kaolin_cam.intrinsics.fov_y) / 180 * math.pi
    proj_matrix = perspective_matrix_colmap(
        znear=1e-2, zfar=1e2, fov_x=fov_x, fov_y=fov_y).transpose(0, 1)

    kaolin_cam2world = kaolin_cam.extrinsics.inv_view_matrix()[0]
    colmap_cam2world = kaolin2colmap_cam2world(kaolin_cam2world)
    colmap_world2cam = colmap_cam2world.inverse().transpose(0, 1)
    colmap_fullproj = colmap_world2cam @ proj_matrix.to(colmap_world2cam)
    return GSCamera(colmap_world2cam, colmap_fullproj,
                    fov_x=fov_x, fov_y=fov_y,
                    image_width=kaolin_cam.width,
                    image_height=kaolin_cam.height,
                    kao_cam=kaolin_cam)


def cam2world_from_angles(azim, elev, radius):
    """
    Create camera to world matrix from angles.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        radius: distance from camera to look_at point
    """
    camera_origins = torch.zeros((azim.shape[0], 3), device=azim.device)

    camera_origins[:, 0:1] = radius * torch.sin(elev) * torch.cos(math.pi-azim)
    camera_origins[:, 2:3] = radius * torch.sin(elev) * torch.sin(math.pi-azim)
    camera_origins[:, 1:2] = radius * torch.cos(elev)

    forward_vectors = normalize(-camera_origins)
    return cam2world_from_direction(forward_vectors, camera_origins)


def cam2world_from_direction(
        forward_vector: torch.Tensor,
        origin: torch.Tensor):
    """Create camera to world matrix from a forward vector and origin.
    Args:
        forward_vector: tensor of shape [batch_size, 3]. The direction the camera is pointing.
        origin: tensor of shape [batch_size, 3]. The camera origin.
    """

    forward_vector = normalize(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics