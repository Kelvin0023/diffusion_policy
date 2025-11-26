import os
import copy
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
            max_train_episodes=None,
    ):
        super().__init__()

        if isinstance(max_train_episodes, str):
            s = max_train_episodes.strip().lower()
            if s in ("", "none", "null", "all"):
                max_train_episodes = None
            else:
                max_train_episodes = int(max_train_episodes)

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["camera_rgb_image", "hand_joint_pos", "ur5_joint_pos", "hand_action", "ur5_action"],
        )

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

        # Initialize the sequence sampler for training data
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

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            "hand_joint_pos": self.replay_buffer["hand_joint_pos"],
            "ur5_joint_pos": self.replay_buffer["ur5_joint_pos"],
            "action": np.concatenate([
                self.replay_buffer["ur5_action"], self.replay_buffer["hand_action"]
            ], axis=-1),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['camera_rgb_image'] = get_image_range_normalizer()
        return normalizer

    def _sample_to_data(self, sample):
        ur5_pos = sample['ur5_joint_pos'].astype(np.float32) # ur5 joint positions (6)
        hand_pos = sample['hand_joint_pos'].astype(np.float32) # hand joint positions (8)
        ur5_action = sample['ur5_action'].astype(np.float32) # ur5 joint pose (7)
        hand_action = sample['hand_action'].astype(np.float32) # hand joint actions (8)
        rgb_image = np.moveaxis(sample['camera_rgb_image'],-1,1)/255
        action = np.concatenate([ur5_action, hand_action], axis=-1)

        data = {
            'obs': {
                'camera_rgb_image': rgb_image,  # T, 3, 640, 480
                'ur5_joint_pos': ur5_pos,  # T, 6
                'hand_joint_pos': hand_pos,  # T, 8
            },
            'action': action # T, 15
        }
        return data

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    dataset_path = "/home/kai/gripper-ros2/collected_data/test_tr01_replay.zarr"
    dataset = PickupImageDataset(
        zarr_path=dataset_path,
        horizon=5,
        max_train_episodes=None,
    )
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Obs keys:", sample['obs'].keys())
    print("camera_rgb_image shape:", sample['obs']['camera_rgb_image'].shape)
    print("ur5_joint_pos shape:", sample['obs']['ur5_joint_pos'].shape)
    print("hand_joint_pos shape:", sample['obs']['hand_joint_pos'].shape)
    print("action shape:", sample['action'].shape)

if __name__ == "__main__":
    test()