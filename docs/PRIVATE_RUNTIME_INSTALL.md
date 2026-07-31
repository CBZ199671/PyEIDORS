# 旧版手动安装说明已停用

PyEIDORS 2.0.0 不再向普通用户分发手动 source zip + binary-cache tar 安装流程。旧流程需要用户自行安装 Nix、启用 flakes、导入缓存和选择 flake profile，容易出现文档与实际包不一致、cache key、PATH 和版本冲突。

当前唯一推荐的 Linux 用户安装入口是三种自解压 `.run` 一键包：

- CPU 通用版；
- NVIDIA SM61 版；
- NVIDIA 现代版。

请阅读：

- 中文：[EASY_INSTALL_LINUX.zh.md](EASY_INSTALL_LINUX.zh.md)
- English: [EASY_INSTALL_LINUX.en.md](EASY_INSTALL_LINUX.en.md)

发布构建入口：

```bash
scripts/release/build_easy_installers.sh 2.0.0 all
```

不要再把旧 `build_binary_cache_bundle.sh` 产生的中间 bundle/tar 直接交给新手用户；它只作为一键包组装的内部构建阶段。
