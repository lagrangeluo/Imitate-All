#!/usr/bin/env python3
"""
创建符合格式要求的示例HDF5文件
基于utils.py中的数据结构分析
"""

import h5py
import numpy as np
import cv2
import os
from typing import List, Dict

def create_sample_hdf5(file_path: str, episode_length: int = 200):
    """
    创建一个示例HDF5文件，符合机器人强化学习数据集的格式
    
    Args:
        file_path: 输出文件路径
        episode_length: episode长度（时间步数）
    """
    
    # 设置随机种子确保可重现
    np.random.seed(42)
    
    # 定义数据维度
    joint_dim = 7  # 机器人关节数量
    action_dim = 7  # 动作维度
    image_height, image_width = 480, 640  # 图像尺寸
    num_cameras = 2  # 相机数量
    camera_names = ["0", "1"]  # 相机名称
    
    print(f"创建示例HDF5文件: {file_path}")
    print(f"Episode长度: {episode_length}")
    print(f"关节维度: {joint_dim}")
    print(f"动作维度: {action_dim}")
    print(f"图像尺寸: {image_height}x{image_width}")
    print(f"相机数量: {num_cameras}")
    
    with h5py.File(file_path, 'w') as root:
        # 设置根属性
        root.attrs["compress"] = True  # 图像压缩标志
        root.attrs["sim"] = False      # 仿真数据标志
        
        # 创建动作数据 - 随机生成机器人动作序列
        print("生成动作数据...")
        action_data = np.random.randn(episode_length, action_dim).astype(np.float32)
        # 添加一些时间相关性，使动作更真实
        for i in range(1, episode_length):
            action_data[i] = action_data[i-1] + 0.1 * np.random.randn(action_dim)
        
        root.create_dataset("/action", data=action_data, dtype="float32")
        
        # 创建观察数据组
        observations_group = root.create_group("/observations")
        
        # 创建关节位置数据 - 与动作相关的关节位置
        print("生成关节位置数据...")
        qpos_data = np.random.randn(episode_length, joint_dim).astype(np.float32)
        # 关节位置与动作相关，但有一些延迟和噪声
        for i in range(1, episode_length):
            qpos_data[i] = qpos_data[i-1] + 0.05 * action_data[i-1] + 0.01 * np.random.randn(joint_dim)
        
        observations_group.create_dataset("qpos", data=qpos_data, dtype="float32")
        
        # 创建关节速度数据（可选）
        print("生成关节速度数据...")
        qvel_data = np.random.randn(episode_length, joint_dim).astype(np.float32)
        # 速度是位置的一阶差分
        for i in range(1, episode_length):
            qvel_data[i] = (qpos_data[i] - qpos_data[i-1]) * 10  # 缩放因子
        
        observations_group.create_dataset("qvel", data=qvel_data, dtype="float32")
        
        # 创建图像数据组
        images_group = observations_group.create_group("images")
        
        # 为每个相机生成图像数据
        print("生成图像数据...")
        for cam_idx, cam_name in enumerate(camera_names):
            print(f"  处理相机 {cam_name}...")
            
            # 生成随机图像序列
            image_list = []
            compressed_len = []
            
            for t in range(episode_length):
                # 创建随机图像，添加一些时间相关性
                if t == 0:
                    # 第一帧：随机图像
                    image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
                else:
                    # 后续帧：基于前一帧添加变化
                    prev_image = image_list[-1]
                    # 解压缩前一帧
                    if len(prev_image) > 0:
                        prev_decompressed = cv2.imdecode(prev_image, cv2.IMREAD_COLOR)
                        if prev_decompressed is not None:
                            # 添加随机变化
                            noise = np.random.randint(-20, 20, prev_decompressed.shape, dtype=np.int16)
                            image = np.clip(prev_decompressed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        else:
                            image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
                    else:
                        image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
                
                # 添加一些视觉特征，使图像更有意义
                # 添加一个移动的圆形
                center_x = int(320 + 200 * np.sin(t * 0.1 + cam_idx * np.pi))
                center_y = int(240 + 150 * np.cos(t * 0.08 + cam_idx * np.pi))
                cv2.circle(image, (center_x, center_y), 30, (255, 0, 0), -1)
                
                # 添加时间戳文本
                cv2.putText(image, f"T:{t:03d}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Cam:{cam_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 压缩图像
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                result, encoded_image = cv2.imencode(".jpg", image, encode_param)
                compressed_image = encoded_image.flatten()
                
                image_list.append(compressed_image)
                compressed_len.append(len(compressed_image))
            
            # 找到最大压缩长度
            max_compressed_len = max(compressed_len)
            
            # 填充所有图像到相同长度
            padded_images = []
            for compressed_image in image_list:
                padded_image = np.zeros(max_compressed_len, dtype=np.uint8)
                padded_image[:len(compressed_image)] = compressed_image
                padded_images.append(padded_image)
            
            # 保存到HDF5
            padded_images_array = np.array(padded_images, dtype=np.uint8)
            images_group.create_dataset(
                cam_name, 
                data=padded_images_array, 
                dtype="uint8", 
                chunks=(1, max_compressed_len)
            )
            
            print(f"    相机 {cam_name} 完成，压缩长度范围: {min(compressed_len)}-{max_compressed_len}")
        
        # 保存压缩长度信息
        compressed_len_dict = {}
        for cam_name in camera_names:
            compressed_len_dict[cam_name] = compressed_len
        
        # 将压缩长度信息作为属性保存
        root.attrs["compressed_len"] = str(compressed_len_dict)
    
    print(f"示例HDF5文件创建完成: {file_path}")
    
    # 验证文件结构
    print("\n验证文件结构:")
    with h5py.File(file_path, 'r') as root:
        print(f"根属性: {dict(root.attrs)}")
        print(f"数据集: {list(root.keys())}")
        print(f"观察数据: {list(root['/observations'].keys())}")
        print(f"图像数据: {list(root['/observations/images'].keys())}")
        
        # 显示数据形状
        print(f"\n数据形状:")
        print(f"动作数据: {root['/action'].shape}")
        print(f"关节位置: {root['/observations/qpos'].shape}")
        print(f"关节速度: {root['/observations/qvel'].shape}")
        for cam_name in camera_names:
            print(f"相机 {cam_name} 图像: {root[f'/observations/images/{cam_name}'].shape}")

def test_read_hdf5(file_path: str):
    """
    测试读取HDF5文件，验证数据格式
    """
    print(f"\n测试读取HDF5文件: {file_path}")
    
    with h5py.File(file_path, 'r') as root:
        # 检查压缩标志
        compressed = root.attrs.get("compress", False)
        print(f"压缩标志: {compressed}")
        
        # 读取动作数据
        action = root["/action"][:]
        print(f"动作数据形状: {action.shape}")
        print(f"动作数据示例 (前5个时间步):\n{action[:5]}")
        
        # 读取关节位置数据
        qpos = root["/observations/qpos"][:]
        print(f"关节位置形状: {qpos.shape}")
        print(f"关节位置示例 (前5个时间步):\n{qpos[:5]}")
        
        # 读取图像数据
        for cam_name in root["/observations/images"].keys():
            images = root[f"/observations/images/{cam_name}"][:]
            print(f"相机 {cam_name} 图像形状: {images.shape}")
            
            # 解压缩第一帧图像
            if compressed and len(images) > 0:
                first_frame_compressed = images[0]
                # 找到非零数据的长度
                non_zero_length = np.where(first_frame_compressed != 0)[0]
                if len(non_zero_length) > 0:
                    actual_length = non_zero_length[-1] + 1
                    actual_compressed = first_frame_compressed[:actual_length]
                    
                    # 解压缩
                    decompressed = cv2.imdecode(actual_compressed, cv2.IMREAD_COLOR)
                    if decompressed is not None:
                        print(f"  第一帧解压缩后形状: {decompressed.shape}")
                        
                        # 保存第一帧图像用于验证
                        output_image_path = f"test_frame_cam_{cam_name}.jpg"
                        cv2.imwrite(output_image_path, decompressed)
                        print(f"  第一帧已保存到: {output_image_path}")
                    else:
                        print(f"  第一帧解压缩失败")
                else:
                    print(f"  第一帧数据为空")

if __name__ == "__main__":
    # 创建示例HDF5文件
    output_file = "sample_episode_0.hdf5"
    create_sample_hdf5(output_file, episode_length=100)
    
    # 测试读取
    test_read_hdf5(output_file)
    
    print(f"\n示例HDF5文件已创建: {output_file}")
    print("该文件符合utils.py中定义的数据格式要求") 