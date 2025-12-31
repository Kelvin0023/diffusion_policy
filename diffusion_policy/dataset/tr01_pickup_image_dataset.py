import os
import copy
import gc
import torch
import numpy as np
from typing import Dict

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class PickupImageDataset(BaseImageDataset):
    def __init__(
            self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            use_rel_actions=False,
            max_train_episodes=None,
            data_freq=20,
            learning_freq=20,
    ):
        super().__init__()

        # Basic sanity checks for frequencies
        if data_freq <= 0 or learning_freq <= 0:
            raise ValueError("data_freq and learning_freq must be positive integers.")

        if data_freq % learning_freq != 0:
            raise ValueError(
                f"data_freq ({data_freq}) must be divisible by learning_freq ({learning_freq}) "
                "to use frame skipping."
            )

        self.data_freq = data_freq
        self.learning_freq = learning_freq

        # Flag for absolute/relative actions
        self.use_rel_actions = use_rel_actions
        if self.use_rel_actions:
            data_keys = ["camera_rgb_image", "hand_joint_pos", "ur5_joint_pos", "ur5_ee_pose",
                         "hand_action_rel", "ur5_action_rel"]
        else:
            data_keys = ["camera_rgb_image", "hand_joint_pos", "ur5_joint_pos", "ur5_ee_pose",
                         "hand_action", "ur5_action"]

        if isinstance(max_train_episodes, str):
            s = max_train_episodes.strip().lower()
            if s in ("", "none", "null", "all"):
                max_train_episodes = None
            else:
                max_train_episodes = int(max_train_episodes)

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=data_keys,
        )

        # Compute the frame skipping factor (how many raw frames per learning step)
        self.skip_factor = max(1, data_freq // learning_freq)

        # Create mask for training and validation episodes
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # Raw sequence length sampled from the buffer BEFORE skipping
        raw_sequence_length = horizon * self.skip_factor

        # Initialize the sequence sampler for training data
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=raw_sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.raw_sequence_length = raw_sequence_length

        # GC-related config
        self.gc_interval = 10
        self._gc_counter = 0

    def __del__(self):
        """Best-effort cleanup when the dataset is destroyed."""
        try:
            del self.replay_buffer
        except AttributeError:
            pass
        try:
            del self.sampler
        except AttributeError:
            pass

        gc.collect()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        raw_sequence_length = self.horizon * self.skip_factor
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=raw_sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        val_set.raw_sequence_length = raw_sequence_length
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        if self.use_rel_actions:
            action = np.concatenate([
                self.replay_buffer["ur5_action_rel"], self.replay_buffer["hand_action_rel"]
            ], axis=-1)
        else:
            action = np.concatenate([
                self.replay_buffer["ur5_action"], self.replay_buffer["hand_action"]
            ], axis=-1)
        data = {
            "hand_joint_pos": self.replay_buffer["hand_joint_pos"],
            "ur5_joint_pos": self.replay_buffer["ur5_joint_pos"],
            "ur5_ee_pose": self.replay_buffer["ur5_ee_pose"],
            "action": action
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['camera_rgb_image'] = get_image_range_normalizer()
        return normalizer

    def _sample_to_data(self, sample):
        ur5_pos = sample['ur5_joint_pos'].astype(np.float32)  # T, 6
        hand_pos = sample['hand_joint_pos'].astype(np.float32)  # T, 8
        ur5_ee_pos = sample['ur5_ee_pose'].astype(np.float32)  # T, 7

        if self.use_rel_actions:
            ur5_action = sample['ur5_action_rel'].astype(np.float32)   # T, 6
            hand_action = sample['hand_action_rel'].astype(np.float32) # T, 8
        else:
            ur5_action = sample['ur5_action'].astype(np.float32)   # T, 7
            hand_action = sample['hand_action'].astype(np.float32) # T, 8

        rgb_image = sample['camera_rgb_image'] / 255.0  # T, 3, H, W
        action = np.concatenate([ur5_action, hand_action], axis=-1)

        data = {
            'obs': {
                'camera_rgb_image': rgb_image,  # T, 3, H, W
                'ur5_joint_pos': ur5_pos,       # T, 6
                'hand_joint_pos': hand_pos,     # T, 8
                'ur5_ee_pose': ur5_ee_pos,      # T, 7
            },
            'action': action  # T, 14 or T, 15
        }
        return data

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample a raw high-frequency sequence
        sample = self.sampler.sample_sequence(idx)

        # Apply frame skipping along the temporal dimension if needed
        if self.skip_factor > 1:
            for key, value in sample.items():
                # Downsample time dimension: (T_raw, ...) -> (T_raw / skip_factor, ...)
                sample[key] = value[::self.skip_factor]

        # At this point, sample has length == self.horizon along time
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        # Help GC by dropping references ASAP
        del sample
        del data

        self._gc_counter += 1
        if self._gc_counter % self.gc_interval == 0:
            gc.collect()

        return torch_data


def test():
    dataset_path = "/home/kai/gripper-ros2/collected_data_pickup_yellow/pickup_yellow_1216.zarr"
    dataset = PickupImageDataset(
        zarr_path=dataset_path,
        horizon=5,
        max_train_episodes=None,
        data_freq=20,
        learning_freq=10,
        use_rel_actions=True,
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Skip factor: {dataset.skip_factor}")

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Obs keys:", sample['obs'].keys())
    print("camera_rgb_image shape:", sample['obs']['camera_rgb_image'].shape)
    print("ur5_joint_pos shape:", sample['obs']['ur5_joint_pos'].shape)
    print("hand_joint_pos shape:", sample['obs']['hand_joint_pos'].shape)
    print("ur5_ee_pose shape:", sample['obs']['ur5_ee_pose'].shape)
    print("action shape:", sample['action'].shape)

    # cleanup
    del sample
    del dataset
    gc.collect()


if __name__ == "__main__":
    test()
