import zarr

filepath = "/home/kai/gripper-ros2/collected_data/test_tr01/test_tr01_replay.zarr"
root = zarr.open(filepath, mode="r")

data = root["data"]
meta = root["meta"]

print("Data arrays:", list(data.array_keys()))

img = data["camera_rgb_image"]
action = data["hand_joint_pos"]
episode_ends = meta["episode_ends"]

print("img shape:", img.shape)
print("action shape:", action.shape)
print("episode_ends shape:", episode_ends.shape)
print("Episode ends:", episode_ends[:10])  # Print first 10 episode ends
