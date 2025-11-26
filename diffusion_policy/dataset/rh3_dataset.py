from typing import Dict
import torch
import numpy as np
import copy
import os
import pickle
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, BaseImageDataset
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from data_management.rh3_replay_buffer import ReplayBuffer4Realworld
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from tqdm import tqdm
import zarr

import psutil
process = psutil.Process(os.getpid())
import gc
import cv2



class ImageDataset(BaseImageDataset):
    def __init__(self,shape_meta: dict, max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape, episodes_path, processed_data_path='rh3_processed_data', 
                 horizon=10, pad_before=0, pad_after=0, n_obs_steps=None,
                n_latency_steps=0, is_train=True, use_rot6d=True, 
                 seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__()

        self.use_rot6d = use_rot6d
        if use_rot6d:
            self.rotation_transformer = RotationTransformer('quaternion','rotation_6d')

        self.shape_meta = shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys=rgb_keys
        self.lowdim_keys=lowdim_keys
        self.action_key='action'
        self.state_ids_key = ['current_variation_idx', 'current_episode_idx', 'current_step']

        self.max_replay_buffer_size = max_replay_buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.RLdataset_image_shape = RLdataset_image_shape
        # self.processed_image_shape = shape_meta['obs'][self.rgb_keys[0]]['shape']
        self.processed_image_shapes = {key: shape_meta['obs'][key]['shape'] for key in self.rgb_keys}
        self.episodes_path = episodes_path
        self.processed_data_path = processed_data_path

        self.replay_buffer = self.load_or_process_data(max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape, episodes_path, processed_data_path, is_train)
        self.is_train = is_train
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        # import ipdb; ipdb.set_trace()

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
    

    def load_or_process_data(self, max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape, episodes_path, processed_data_path, is_train):
        file_name = 'train_replay_buffer.zarr' if is_train else 'test_replay_buffer.zarr'
        file_path = os.path.join(self.processed_data_path, file_name)

        if obs_dim ==19 or obs_dim == 17 or obs_dim == 10 or obs_dim == 25:
            obs_dim = 8
        if action_dim == 10:
            action_dim = 8
        if os.path.exists(file_path):
            print(f'Loading preprocessed {"train" if is_train else "test"} data from {self.processed_data_path}')
            replay_buffer = ReplayBuffer.copy_from_path(file_path)
            print('Preprocessed data loaded.')
        else:
            print('Processing data...')
            store = zarr.DirectoryStore(file_path)
            root = zarr.group(store, overwrite=True)
            data_group = root.require_group('data', overwrite=True)
            meta_group = root.require_group('meta', overwrite=True)

            total_steps = 0
            episode_ends = []
            
            data_arrays = {}
            for key in self.lowdim_keys + self.rgb_keys + [self.action_key]:
                dtype = np.uint8 if key in self.rgb_keys else np.float32

                # Determine data shape based on key type
                if key in self.rgb_keys:
                    data_shape = tuple(self.processed_image_shapes[key])  # e.g., (3, 128, 128)
                elif key in self.lowdim_keys:
                    data_shape = tuple(self.shape_meta['obs'][key]['shape'])  # e.g., (4,) or (3,)
                elif key == self.action_key:
                    data_shape = (self.action_dim,)  # action_dim
                else:
                    raise ValueError(f"Unknown key: {key}")

                shape = (0,) + data_shape  # Start with size 0
                chunks = (1,) + data_shape  # Optimal chunk size for appending one sample at a time

                data_arrays[key] = data_group.require_dataset(
                    name=key,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    compressor=None,
                    overwrite=True,
                )

            rlbench_buffer = ReplayBuffer4Realworld(max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape)
            rlbench_buffer.load_rlbench_data(episodes_path)

            observations = rlbench_buffer._observations  # Shape: (N, obs_dim)
            actions = rlbench_buffer._next_observations            # Shape: (N, action_dim)
            image_observations_dict = rlbench_buffer._image_observations  # Shape: (N, H, W, C)
            terminals = rlbench_buffer._terminals        # Shape: (N,)

            N = len(observations)

            
            if not is_train:
                # Use only 20% of the data for testing
                N = int(0.2 * N)
                print(f'Using only {N} episodes for testing.')

            episode_data = {key: [] for key in self.lowdim_keys + self.rgb_keys + [self.action_key]}

            skip_current_episode = False
            for t in tqdm(range(N), desc='Processing data'):
                obs = observations[t]
                obs_dict = {}
                idx = 0

                for key in self.lowdim_keys:
                    if key.endswith('eef_quat'):
                        shape = (4,)
                    else:
                        shape = self.shape_meta['obs'][key]['shape']         
                    key_obs_dim = np.prod(shape)
                    obs_dict[key] = obs[idx:idx + key_obs_dim]
                    idx += key_obs_dim
                
                if self.use_rot6d:
                    for key in self.lowdim_keys:
                        if key.endswith('eef_quat'):
                            quat = obs_dict[key]
                            if np.all(np.abs(quat) < 0.001):
                                skip_current_episode = True
                                break
                            rot_6d = self.rotation_transformer.forward(quat)
                            obs_dict[key] = rot_6d

                for key in self.rgb_keys:
                    if key in image_observations_dict:
                        try:
                            img_obs = image_observations_dict[key][t]
                            # Move channels if necessary
                            if img_obs.shape[-1] == 3:
                                img_obs = np.moveaxis(img_obs, -1, 0)  # H, W, C -> C, H, W
                            expected_shape = tuple(self.processed_image_shapes[key])
                            assert img_obs.shape == expected_shape, f'Image shape mismatch for {key}: {img_obs.shape} vs {expected_shape}'

                            # **Assign img_obs to obs_dict[key] before the check**
                            obs_dict[key] = img_obs

                            # Perform black pixel ratio check
                            black_pixels = np.sum(img_obs == 0)
                            total_pixels = img_obs.size
                            black_pixel_ratio = black_pixels / total_pixels
                            if black_pixel_ratio > 0.3:  # Adjust threshold as needed
                                # print(f"Skipping episode due to high black pixel ratio in '{key}' at timestep {t}")
                                skip_current_episode = True
                                break  # Exit the loop over rgb_keys

                        except Exception as e:
                            # print(f"Exception occurred while processing '{key}' at timestep {t}: {e}")
                            skip_current_episode = True
                            break

                    else:
                        print(f"Image observations for key {key} not found in buffer.")
                        skip_current_episode = True
                        break
                
                action = actions[t]
                # there is no quat in the action
                # if self.use_rot6d:
                #     quat_idx = 3
                #     quat = action[quat_idx:quat_idx + 4]
                #     rot_6d = self.rotation_transformer.forward(quat)
                #     action = np.concatenate([action[:quat_idx], rot_6d, action[quat_idx + 4:]])

                if skip_current_episode:
                # Reset episode data
                    episode_data = {key: [] for key in self.lowdim_keys + self.rgb_keys + [self.action_key]}
                    skip_current_episode = False
                    continue           

                for key in self.lowdim_keys + self.rgb_keys:
                    episode_data[key].append(obs_dict[key])
                episode_data[self.action_key].append(action)
                # Check for episode termination
                if terminals[t] or t == N - 1:
                    if len(episode_data[self.action_key]) > 0:
                        # Convert lists to arrays
                        for key in episode_data:
                            episode_data[key] = np.array(episode_data[key])
                        # Write data to Zarr datasets
                        episode_length = len(episode_data[self.action_key])
                        for key in self.lowdim_keys + self.rgb_keys + [self.action_key]:
                            data_arrays[key].append(episode_data[key])

                        total_steps += episode_length
                        episode_ends.append(total_steps)

                        # Clear episode data to free memory
                        episode_data = {key: [] for key in self.lowdim_keys + self.rgb_keys + [self.action_key]}
                    else:
                        print(f"Skipped an episode ending at timestep {t}")
                        skip_current_episode = False
                    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
                    print(f"Memory usage at timestep {t}: {memory_usage:.2f} MB")

                    # Collect garbage to free memory
                    import gc
                    gc.collect()

            meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None, overwrite=True)

            replay_buffer = ReplayBuffer(root)
            print('Data processing completed and saved to Zarr store.')

        return replay_buffer
    
    def get_validation_dataset(self):
        if self.is_train:

            # configure dataset
            return ImageDataset(
                self.shape_meta,
                self.max_replay_buffer_size,
                self.obs_dim,
                self.action_dim,
                self.RLdataset_image_shape,
                self.episodes_path,
                processed_data_path = self.processed_data_path,
                horizon=self.horizon,  
                pad_before=self.pad_before, 
                pad_after=self.pad_after, 
                n_obs_steps = self.n_obs_steps,
                n_latency_steps = self.n_latency_steps,
                is_train=False,
                use_rot6d=True, 
            )
        else:
            return self


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer[self.action_key])
        this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

        normalizer[self.action_key] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported: {key}')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)

        obs_dict = dict()
        for key in self.rgb_keys:
            img = data[key][T_slice].astype(np.float32) / 255.

            if img.ndim == 4 and img.shape[-1] == 3:
                img = np.moveaxis(img, -1, -3)
            
            obs_dict[key] = torch.from_numpy(img)
            del data[key]

        for key in self.lowdim_keys:
            obs = data[key][T_slice].astype(np.float32)
            obs_dict[key] = torch.from_numpy(obs)
            del data[key]

        action = data[self.action_key].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        del data[self.action_key]

        torch_data = {
            'obs': obs_dict,                       
            'action': torch.from_numpy(action)    
        }
        return torch_data

        # obs_dict = dict()
        # for key in self.rgb_keys:
        #     img  = data[key].astype(np.float32) / 255.
        #     if img.shape[-1] == 3:
        #         img = np.moveaxis(img, -1, -3)
        #     obs_dict[key] = torch.from_numpy(img)
        #     obs_dict[key] = obs_dict[key][0:1, ...]
        #     # T,C,H,W
        #     del data[key]

        # for key in self.lowdim_keys:
        #     obs_dict[key] = torch.from_numpy(data[key].astype(np.float32))
        #     obs_dict[key] = obs_dict[key][0:1, ...]
        #     del data[key]

        # # torch_data = dict_apply(data, torch.from_numpy)
        # torch_data = {
        #     'obs': obs_dict,
        #     'action': torch.from_numpy(data[self.action_key].astype(np.float32))
        # }
        # return torch_data

    
class HumanExpertDataset(BaseImageDataset):
    def __init__(self,shape_meta: dict, max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape, episodes_path, processed_data_path='rh3_processed_data', 
                 horizon=10, pad_before=0, pad_after=0, n_obs_steps=None,
                n_latency_steps=0, is_train=True, use_rot6d=True, 
                 seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__()

        self.use_rot6d = use_rot6d
        if use_rot6d:
            self.rotation_transformer = RotationTransformer('quaternion','rotation_6d')

        self.shape_meta = shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys=rgb_keys
        self.lowdim_keys=lowdim_keys
        self.action_key='action'
        self.state_ids_key = ['current_variation_idx', 'current_episode_idx', 'current_step']

        self.max_replay_buffer_size = 1674  #max_replay_buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.RLdataset_image_shape = RLdataset_image_shape
        # self.processed_image_shape = shape_meta['obs'][self.rgb_keys[0]]['shape']
        self.processed_image_shapes = {key: shape_meta['obs'][key]['shape'] for key in self.rgb_keys}
        self.episodes_path = episodes_path
        self.processed_data_path = processed_data_path

        self.replay_buffer = self.load_or_process_data(self.max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape, episodes_path, processed_data_path, is_train)
        self.is_train = is_train
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
    

    def load_or_process_data(self, max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape, episodes_path, processed_data_path, is_train):
        file_name = 'train_replay_buffer.zarr' if is_train else 'test_replay_buffer.zarr'
        file_path = os.path.join(self.processed_data_path, file_name)

        if obs_dim ==19 or obs_dim == 17 or obs_dim == 10 or obs_dim == 25:
            obs_dim = 8
        if action_dim == 10:
            action_dim = 8
        if os.path.exists(file_path):
            print(f'Loading preprocessed {"train" if is_train else "test"} data from {self.processed_data_path}')
            replay_buffer = ReplayBuffer.copy_from_path(file_path)
            print('Preprocessed data loaded.')
        else:
            print('Processing data...')
            store = zarr.DirectoryStore(file_path)
            root = zarr.group(store, overwrite=True)
            data_group = root.require_group('data', overwrite=True)
            meta_group = root.require_group('meta', overwrite=True)

            total_steps = 0
            episode_ends = []
            
            data_arrays = {}
            for key in self.lowdim_keys + self.rgb_keys + [self.action_key]:
                dtype = np.uint8 if key in self.rgb_keys else np.float32

                # Determine data shape based on key type
                if key in self.rgb_keys:
                    data_shape = tuple(self.processed_image_shapes[key])  # e.g., (3, 128, 128)
                elif key in self.lowdim_keys:
                    data_shape = tuple(self.shape_meta['obs'][key]['shape'])  # e.g., (4,) or (3,)
                elif key == self.action_key:
                    data_shape = (self.action_dim,)  # action_dim
                else:
                    raise ValueError(f"Unknown key: {key}")

                shape = (0,) + data_shape  # Start with size 0
                chunks = (1,) + data_shape  # Optimal chunk size for appending one sample at a time

                data_arrays[key] = data_group.require_dataset(
                    name=key,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    compressor=None,
                    overwrite=True,
                )

            rlbench_buffer = ReplayBuffer4Realworld(max_replay_buffer_size, obs_dim, action_dim, RLdataset_image_shape)
            rlbench_buffer.load_rlbench_data(episodes_path)

            observations = rlbench_buffer._observations  # Shape: (N, obs_dim)
            actions = rlbench_buffer._next_observations            # Shape: (N, action_dim)
            image_observations_dict = rlbench_buffer._image_observations  # Shape: (N, H, W, C)
            terminals = rlbench_buffer._terminals        # Shape: (N,)

            N = len(observations)

            if not is_train:
                # Use only 20% of the data for testing
                N = int(0.2 * N)
                print(f'Using only {N} episodes for testing.')

            episode_data = {key: [] for key in self.lowdim_keys + self.rgb_keys + [self.action_key]}

            skip_current_episode = False
            for t in tqdm(range(N), desc='Processing data'):
                obs = observations[t]
                obs_dict = {}
                idx = 0

                for key in self.lowdim_keys:
                    if key.endswith('eef_quat'):
                        shape = (4,)
                    else:
                        shape = self.shape_meta['obs'][key]['shape']         
                    key_obs_dim = np.prod(shape)
                    obs_dict[key] = obs[idx:idx + key_obs_dim]
                    idx += key_obs_dim
                
                if self.use_rot6d:
                    for key in self.lowdim_keys:
                        if key.endswith('eef_quat'):
                            quat = obs_dict[key]
                            if np.all(np.abs(quat) < 0.001):
                                skip_current_episode = True
                                break
                            rot_6d = self.rotation_transformer.forward(quat)
                            obs_dict[key] = rot_6d

                for key in self.rgb_keys:
                    if key in image_observations_dict:
                        try:
                            img_obs = image_observations_dict[key][t]

                            if img_obs.shape[-1] == 3:
                                img_obs = np.moveaxis(img_obs, -1, 0)  # H, W, C -> C, H, W
                            
                            expected_shape = tuple(self.processed_image_shapes[key])
                            assert img_obs.shape == expected_shape, \
                                f"Image shape mismatch for {key}: {img_obs.shape} vs {expected_shape}"

                            if (
                                np.issubdtype(img_obs.dtype, np.floating)  
                                and img_obs.min() >= 0.0
                                and img_obs.max() <= 1.0
                            ):
                                img_obs = (img_obs * 255).astype(np.uint8)

                            obs_dict[key] = img_obs

                            black_pixels = np.sum(img_obs == 0)
                            total_pixels = img_obs.size
                            black_pixel_ratio = black_pixels / total_pixels

                            if black_pixel_ratio > 0.3:  
                                skip_current_episode = True
                                break  

                        except Exception as e:
                            skip_current_episode = True
                            break

                    else:
                        print(f"Image observations for key {key} not found in buffer.")
                        skip_current_episode = True
                        break
                
                action = actions[t]
                # there is no quat in the action
                # if self.use_rot6d:
                #     quat_idx = 3
                #     quat = action[quat_idx:quat_idx + 4]
                #     rot_6d = self.rotation_transformer.forward(quat)
                #     action = np.concatenate([action[:quat_idx], rot_6d, action[quat_idx + 4:]])

                if skip_current_episode:
                # Reset episode data
                    episode_data = {key: [] for key in self.lowdim_keys + self.rgb_keys + [self.action_key]}
                    skip_current_episode = False
                    continue           

                for key in self.lowdim_keys + self.rgb_keys:
                    episode_data[key].append(obs_dict[key])
                episode_data[self.action_key].append(action)
              
                # Check for episode termination
                if terminals[t] or t == N - 1:
                    if len(episode_data[self.action_key]) > 0:
                        # Convert lists to arrays
                        for key in episode_data:
                            episode_data[key] = np.array(episode_data[key])
                        # Write data to Zarr datasets
                        episode_length = len(episode_data[self.action_key])
                        for key in self.lowdim_keys + self.rgb_keys + [self.action_key]:
                            try:
                                data_arrays[key].append(episode_data[key])
                            except Exception as e:
                                print(f"Error appending data for key {key}: {e}")
                                import ipdb; ipdb.set_trace()

                        total_steps += episode_length
                        episode_ends.append(total_steps)

                        # Clear episode data to free memory
                        episode_data = {key: [] for key in self.lowdim_keys + self.rgb_keys + [self.action_key]}
                    else:
                        print(f"Skipped an episode ending at timestep {t}")
                        skip_current_episode = False
                    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
                    print(f"Memory usage at timestep {t}: {memory_usage:.2f} MB")

                    # Collect garbage to free memory
                    import gc
                    gc.collect()

            meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None, overwrite=True)
            replay_buffer = ReplayBuffer(root)
            print('Data processing completed and saved to Zarr store.')
        return replay_buffer

    def get_validation_dataset(self):
        if self.is_train:
            # configure dataset
            return HumanExpertDataset(
                self.shape_meta,
                self.max_replay_buffer_size,
                self.obs_dim,
                self.action_dim,
                self.RLdataset_image_shape,
                self.episodes_path,
                processed_data_path = self.processed_data_path,
                horizon=self.horizon,  
                pad_before=self.pad_before, 
                pad_after=self.pad_after, 
                n_obs_steps = self.n_obs_steps,
                n_latency_steps = self.n_latency_steps,
                is_train=False,
                use_rot6d=True, 
            )
        else:
            return self


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer[self.action_key])
        this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

        normalizer[self.action_key] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported: {key}')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)

        obs_dict = dict()
        for key in self.rgb_keys:
            img = data[key][T_slice].astype(np.float32) / 255.

            if img.ndim == 4 and img.shape[-1] == 3:
                img = np.moveaxis(img, -1, -3)
            
            obs_dict[key] = torch.from_numpy(img)
            del data[key]

        for key in self.lowdim_keys:
            obs = data[key][T_slice].astype(np.float32)
            obs_dict[key] = torch.from_numpy(obs)
            del data[key]

        action = data[self.action_key].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        del data[self.action_key]

        torch_data = {
            'obs': obs_dict,                       
            'action': torch.from_numpy(action)    
        }
        return torch_data

