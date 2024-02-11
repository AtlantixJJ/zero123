import torch, copy, csv, cv2, math, json, random, os, sys
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict
from pathlib import Path
from torch.utils.data import Dataset
from omegaconf import DictConfig, ListConfig
from torchvision import transforms
from einops import rearrange
from os.path import join as osj
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Compose, Resize, Lambda

from .camera_helper import matrix_to_quaternion
from ldm.util import instantiate_from_config
from datasets import load_dataset


# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)


def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)


class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([ToTensor(),
                                Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class FaceDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, ann_fpath, sample_acam=-1, resolution=512, batch_size=1, total_view=4, cond_radius=False, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.ann_fpath = ann_fpath
        self.sample_acam = sample_acam
        self.resolution = resolution
        self.batch_size = batch_size
        self.total_view = total_view
        self.cond_radius = cond_radius
        self.num_workers = num_workers

        self.transforms = Compose([
            Resize(self.resolution),
            ToTensor(),
            Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

    def train_dataloader(self):
        dataset = MultiViewDataset(
            root_dir=self.root_dir,
            ann_fpath=self.ann_fpath,
            cond_radius=self.cond_radius,
            transforms=self.transforms)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = MultiViewDataset(
            root_dir=self.root_dir,
            ann_fpath=self.ann_fpath,
            cond_radius=self.cond_radius,
            transforms=self.transforms)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)
    
    def test_dataloader(self):
        dataset = MultiViewDataset(
            root_dir=self.root_dir,
            ann_fpath=self.ann_fpath,
            cond_radius=self.cond_radius,
            transforms=self.transforms)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)


def imread_pil(fpath, size=None):
    img = Image.open(open(fpath, "rb")).convert("RGB")
    if size is not None:
        img = img.resize(size, resample=Image.Resampling.BILINEAR)
    return img


class MultiViewDataset(torch.utils.data.Dataset):
    """Multi-view dataset."""

    def __init__(self, root_dir, ann_fpath, cond_radius=False, sample_acam=-1, xflip=False, transforms=None):
        self.ann_fpath = ann_fpath
        self.root_dir = root_dir
        self.sample_acam = sample_acam # sample around camera, quaternion threshold
        self.xflip = xflip
        self.cond_radius = cond_radius # whether to use stable zero123 setup
        self.scenes = torch.load(ann_fpath, map_location='cpu')
        self.scene_names = list(self.scenes.keys())
        self.scene_names.sort()
        indice = []
        for i, k in enumerate(self.scene_names):
            s = self.scenes[k]
            n_cam = len(s['_camdata_image_path'])
            indice.append(np.stack([
                i * np.ones((n_cam,)),
                np.arange(n_cam)], 1))
        self.sample_indice = np.concatenate(indice, 0).astype('int32')

        self.transforms = transforms

    def __len__(self):
        return len(self.sample_indice)

    def get_camera_dict(self, scene_prefix, cam_dict, cam_idx):
        """Get camera dictionary."""
        dic = {k: v[cam_idx] for k, v in cam_dict.items()
            if k.startswith('_camdata_') and v[cam_idx] is not None}

        if '_camdata_image_path' in cam_dict:
            image_path = osj(scene_prefix, cam_dict['_camdata_image_path'][cam_idx][1:])
            dic['_camdata_image'] = self.transforms(imread_pil(image_path))

        if '_camdata_depth_path' in cam_dict:
            depth_path = osj(scene_prefix, cam_dict['_camdata_depth_path'][cam_idx][1:])
            dic['_camdata_depth'] = torch.from_numpy(np.load(depth_path))

        if '_camdata_mask_path' in cam_dict:
            mask_path = osj(scene_prefix, cam_dict['_camdata_mask_path'][cam_idx][1:])
            dic['_camdata_mask'] = self.transforms(imread_pil(mask_path))[:1]
        return dic

    def __getitem__(self, idx):
        """
        Returns:
            scene_idx: int
            camera_idx: int
            camera: List of Camera object.
            ref_image: torch.Tensor [3, H, W] in [0, 1]
            gt_image: torch.Tensor [3, H, W] in [0, 1]
            camera_extent: float
        """
        scene_idx, cam_idx = self.sample_indice[idx]
        scene_name = self.scene_names[scene_idx]
        scene_prefix = osj(self.root_dir, scene_name)

        cam_dict = self.scenes[scene_name]
        scene_image_fps = cam_dict['_camdata_image_path']

        n_cam = len(scene_image_fps)

        # random sample a camera as reference
        refnear_cam_idx = -1
        if self.sample_acam > 0:
            cam2quat = lambda idx: matrix_to_quaternion(
                torch.linalg.inv(
                    cam_dict['_camdata_world_view_transform'][idx].T)[:3, :3])
            gt_quat = cam2quat(cam_idx)
            indices = np.arange(n_cam)
            np.random.shuffle(indices)
            for i in range(n_cam):
                ref_quat = cam2quat(indices[i])
                dist = (gt_quat - ref_quat).norm()
                if dist < self.sample_acam and refnear_cam_idx != cam_idx:
                    refnear_cam_idx = indices[i]
                    break
            if refnear_cam_idx == -1:
                refnear_cam_idx = indices[0]
                print("Warning: no camera found for sampling!")
            refnear_camera = self.get_camera_dict(
                scene_prefix, cam_dict, refnear_cam_idx)
        
        ref_cam_idx = np.random.randint(n_cam)
        while ref_cam_idx in [cam_idx, refnear_cam_idx]:
            ref_cam_idx = np.random.randint(n_cam)

        ref_camera = self.get_camera_dict(scene_prefix, cam_dict, ref_cam_idx)
        tar_camera = self.get_camera_dict(scene_prefix, cam_dict, cam_idx)

        data = {
            'scene_idx': scene_idx,
            'camera_idx': cam_idx
        }
        #m = ref_camera['_camdata_mask']
        #data['image_target'] = (ref_camera['_camdata_image'] * m) + 1 - m # white background
        #m = tar_camera['_camdata_mask']
        #data['image_cond'] = (tar_camera['_camdata_image'] * m) + 1 - m # white background
        data['image_target'] = ref_camera['_camdata_image']
        data['image_cond'] = tar_camera['_camdata_image']
        data["T"] = self.get_T(
            torch.inverse(ref_camera['_camdata_world_view_transform'].T),
            torch.inverse(tar_camera['_camdata_world_view_transform'].T))

        return data
    
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_cam2world, cond_cam2world):
        M = torch.Tensor([ # conversion from COLMAP format to blender format
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]]).to(target_cam2world)
        
        #R, T = target_RT[:3, :3], target_RT[:3, -1]
        #T_target = M @ -R.T @ T #-R.T @ T #M @ -R.T @ T

        #R, T = cond_RT[:3, :3], cond_RT[:3, -1]
        #T_cond = M @ -R.T @ T #-R.T @ T #M @ -R.T @ T
        T_cond = M @ cond_cam2world[:3, -1]
        T_target = M @ target_cam2world[:3, -1]

        theta_cond, azimuth_cond, radius_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, radius_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_radius = radius_target - radius_cond

        d_T = torch.tensor([
            d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()),
            d_radius.item() if self.cond_radius else theta_cond.item()])
        #print(f'cond radius {radius_cond.item():.3f} target radius {radius_target.item():.3f}')
        #print(T_cond)
        #print(T_target)
        #print(d_T)
        return d_T


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([
            d_theta.item(),
            math.sin(d_azimuth.item()),
            math.cos(d_azimuth.item()),
            d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random


class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
