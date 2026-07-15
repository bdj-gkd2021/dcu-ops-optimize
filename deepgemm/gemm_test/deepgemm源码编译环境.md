# 环境搭建

## 基本新型

海光BW1000

算力 480TFLOPS

带宽 1\.8T

显存 64G

服务器的操作系统: SugonOS 8\.9



docker调用时，推荐挂载/sys/kernel/debug路径，不然，容器内无法获取到dcu进程信息



DTK 26\.04 文档: https://download\.sourcefind\.cn:65024/1/main/latest/Document

DAS 1\.8 文档: https://das\.sourcefind\.cn:55011/portal/\#/docs/DAS%E4%BB%8B%E7%BB%8D



可以用`ibstat`命令查看IB网卡

## 创建容器

```Bash
docker run -it --name txz_masked_ll \
--privileged \
--pid=host \
--network=host \
--ipc=host \
--device=/dev/kfd \
--device=/dev/mkfd \
--device=/dev/dri \
-v /opt/hyhal:/opt/hyhal \
-v /sys/kernel/debug:/sys/kernel/debug \
-v /zkjh:/zkjh \
-v /home/o_tangxiangzhu \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260410 \
/bin/bash
```



```Bash
docker run -it --name zkjh-sglang-minimax-opt2 \
--privileged \
--pid=host \
--network=host \
--ipc=host \
--device=/dev/kfd \
--device=/dev/mkfd \
--device=/dev/dri \
-v /opt/hyhal:/opt/hyhal \
-v /sys/kernel/debug:/sys/kernel/debug \
-v /zkjh:/zkjh \
-v /data:/data \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260410 \
/bin/bash
```



根据海光给的\.whl来更新Python包 \(\.whl包已下载至`/zkjh/drivers/`目录）

```Bash
pip install /zkjh/drivers/lmslim-0.3.1+das.opt3.dtk2604.20260121.g3529a6c9-cp310-cp310-manylinux_2_28_x86_64.whl
pip install /zkjh/drivers/mooncake_transfer_engine-0.3.7.post2+das.opt1.dtk2604-cp310-cp310-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl
pip install aiter -i http://42.228.13.241:666/nightly/dtk2604/ --trusted-host 42.228.13.241
```

天龙网卡安装的deepep版本

```Bash
pip install /zkjh/drivers/deep_ep-1.0.0+das.opt1.dtk2604.torch290.shca.2604131629.g766b17-py3-none-any.whl
```

MLNX网卡安装的deepep版本

```Bash
pip install pip install /zkjh/drivers/deep_ep-1.0.0+das.opt1.dtk2604.torch290.2606011725.gdcf58e-cp310-cp310-manylinux_2_28_x86_64.whl
```

#### 容器中配置天龙网卡软件栈 （MLNX无需配置）

参考[天龙网卡插件安装补充参考](https://lhui08qsvi.feishu.cn/wiki/Di9vwiGMiiH7kOk1nthcjWH7nZc?from=from_copylink)

1. 下载天龙IB网卡软件栈 \(**NOTE: 已下载好**，在`/zkjh/drivers/ib_plugin` 目录\)

```Plain Text
#下载linux快传客户端，若有无需重复下载
wget https://zzefile.scnet.cn:65011/efile/share/fasttrans-client.tar.gz

#下载执行脚本
rayfile-c -a zzefile.scnet.cn -P 65012 -u jsyadmin -w 9380ec0f4b92b4ccb1-f638-44e3-b420-3c52876c6bc0 -no-meta -symbolic-links follow -retry 10 -retrytimeout 30 -o download -s '/L4/yongdan/project/GLM5-test/mlxtoshca.sh' -d <请输入下载目标全路径并替换尖括号及本内容>

#下载驱动安装包
rayfile-c -a zzefile.scnet.cn -P 65012 -u jsyadmin -w 9380ec0f4b0e868171-0d49-4ff3-9fa3-5618bb17dd91 -no-meta -symbolic-links follow -retry 10 -retrytimeout 30 -o download -s '/L4/yongdan/shca-tools_2.500.4.B059-Ubuntu22.04_amd64.deb' -d <请输入下载目标全路径并替换尖括号及本内容>

#为了以防万一和多镜像使用，建议将mlxtoshca.sh脚本中最后一行删除安装包的操作去掉
```

2. 下载rccl相关库和拓扑 \(**NOTE:**  **已下载好**\)

- 拓扑文件在`/zkjh/drivers/ib_plugin/topo_lib/built-in-508-topo-input-tj-default.xml`

- rccl插件在`/zkjh/drivers/ib_plugin/topo_lib/lib`目录

```Bash
curl -f -C - -o topo_lib.tar.gz https://zzefile.scnet.cn:65011/efile/s/d/anN5YWRtaW4=/8eb6de1325954d9f 
```

3. 安装软件栈并设置软连接

```Bash
# 安装IB网卡软件栈
cd /zkjh/drivers/ib_plugin
bash mlxtoshca.sh
# 设置软链接
cd /zkjh/drivers/ib_plugin/topo_lib/lib
ln -s librccl-net-shca.so.0.0.0 librccl-net-shca.so
ln -s librccl-net-shca.so.0.0.0 librccl-net-shca.so.0
```

装好后`ibv_devinfo`命令可以看见IB设备



## 源码安装SGLang 

~~把/zkjh/repos/sglang复制到自己的目录里，然后执行~~~~`git reset --hard && git clean -xdf && git checkout 0.5.10rc0_dev_ep`~~~~。~~

```Bash
#无权限
#git clone https://developer.sourcefind.cn/codes/chenlong/sglang

cp -r  /zkjh/repos/OpenDAS/sglang ./
cd sglang
~~git checkout v0.5.10rc0_dev_ep_sp~~
git checkout wangrui
```

源码安装:

```Plain Text
pip install -r requirements_dcu.txt
pip install pycountry
cd sgl-kernel
#注意先注释 setup_hip.py 的time哪一行
python setup_hip.py install
cd ..
pip install -e "python[all_hip]" --no-deps --no-build-isolation --no-index
```

## 源码安装DeepGEMM

```Bash
git clone https://developer.sourcefind.cn/codes/chenlong/deepgemm
cp -r  /zkjh/repos/OpenDAS/deepgemm ./

#修改.h相关文件得删除这个再编译才能得到最新结果
rm -rf build dist deepgemm.egg-info

#删除现在安装的deepgemm版本
rm -rf /usr/local/lib/python3.10/dist-packages/deepgemm
rm -rf /usr/local/lib/python3.10/dist-packages/deepgemm-2.1.0+das.dtk2604.e52469a.dist-info
rm -rf /usr/local/lib/python3.10/dist-packages/deepgemm-*.egg-info

cd improve_host
git checkout improve_host



MAX_JOBS=32 PYTORCH_ROCM_ARCH=gfx936 python3 setup.py bdist_wheel

PYTORCH_ROCM_ARCH=gfx936 python3 setup.py bdist_wheel
#修改成对应的包
pip install --no-deps --force-reinstall ./dist/deepgemm-2.1.0+das.dtk2604.e52469a-cp310-cp310-linux_x86_64.whl

#有时候必须这样安装
cd /workspace/deepgemm
python3 -m pip install \
  --force-reinstall \
  --no-deps \
  dist/deepgemm-2.1.0+das.dtk2604.2dcd5e8-cp310-cp310-linux_x86_64.whl

#非源码编译直接安装整包deepgemm
pip install deepgemm -i http://42.228.13.241:666/nightly/dtk2604/ --trusted-host 42.228.13.241
```



## Deepgemm版本说明

```
deepgemm-2.1.0+das.dtk2604.e52469a-cp310-cp310-linux_x86_64.whl --> masked修复版

deepgemm-2.1.0+das.dtk2604.masked_ll.e52469a-cp310-cp310-linux_x86_64.whl --> masked_ll修复版

deepgemm-2.1.0+das.dtk2604.masked_ll_new.e52469a-cp310-cp310-linux_x86_64.whl -->masked_ll兼容N=384,K=192,BLOCK_N从256改为128

deepgemm-2.1.0+das.dtk2604.masked_ll_new2.e52469a-cp310-cp310-linux_x86_64.whl --> masked_ll在new基础上兼容CU=80
```



