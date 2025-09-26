# PyEidors

基于FEniCS的电阻抗成像(EIT)正逆问题求解系统 - Python版本的EIDORS

## 环境说明

本项目基于Docker环境开发，使用以下核心组件：

- **FEniCS**: ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
- **CUQIpy**: 1.3.0
- **CUQIpy-FEniCS**: 0.8.0  
- **PyTorch**: 2.7.1+cu128 (GPU支持)
- **Python**: 3.10+ (通过Docker提供)

以下的安装方法是在使用fenics的Docker镜像的基础上手动安装剩余的依赖。
也可以使用[Dockerfile](Dockerfile)来构建一个完整的镜像。
或者使用我们提供的完整环境镜像来使用。
## Docker环境设置

```bash
# 启动容器
docker run -ti \
  --gpus all \
  --shm-size=24g \     
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --cpus=20 \            
  --memory=28g \  
  -v "D:\workspace\PyEIDORS:/root/shared" \   
  -w /root/shared \
  --name pyeidors \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

# （可选）安装中文字体依赖
apt-get update && apt-get install ttf-wqy-zenhei
# 安装CUQIpy和CUQIpy-FEniCS
pip install cuqipy cuqipy-fenics

# 创建虚拟环境
python3 -m venv /opt/final_venv --system-site-packages
source /opt/final_venv/bin/activate

# 安装PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128