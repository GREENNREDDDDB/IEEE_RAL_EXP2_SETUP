# IEEE_RAL_EXP2_SETUP

> - 从已配好的环境中创建新环境：`conda env create -f environment.yml`
> - 按照下文安装realsense sdk
> - [下载最新fireware并用viewer安装]()https://github.com/IntelRealSense/librealsense/releases/tag/v2.55.1)
> - 完成片内标定

## 1 Basic env install

### 1.1 yolov8

```
## create virtual env
conda create -n ieee_ral
conda install python==3.11
## install yolo package
pip install ultralytics
```

### 1.2 Realsense

#### 1.2.1 Realsense SDK 2.0 install

- Register the server's public key:

```
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
```

- Make sure apt HTTPS support is installed:
  `sudo apt-get install apt-transport-https`
- Add the server to the list of repositories:

```
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
```

- Install the libraries (see section below if upgrading packages):`sudo apt-get install librealsense2-dkmssudo apt-get install librealsense2-utils`The above two lines will deploy librealsense2 udev rules, build and activate kernel modules, runtime library and executable demos and tools.
- Optionally install the developer and debug packages:
  `sudo apt-get install librealsense2-dev`
  `sudo apt-get install librealsense2-dbg`
  With `dev` package installed, you can compile an application with **librealsense** using `g++ -std=c++11 filename.cpp -lrealsense2` or an IDE of your choice.

Reconnect the Intel RealSense depth camera and run: `realsense-viewer` to verify the installation.

Verify that the kernel is updated :
`modinfo uvcvideo | grep "version:"` should include `realsense` string

#### 1.2.2 Upgrading the Packages:

Refresh the local packages cache by invoking:
  `sudo apt-get update`

Upgrade all the installed packages, including `librealsense` with:
  `sudo apt-get upgrade`

To upgrade selected packages only a more granular approach can be applied:
  `sudo apt-get --only-upgrade install <package1 package2 ...>`
  E.g:
  `sudo apt-get --only-upgrade install  librealsense2-utils librealsense2-dkms`

#### **1.2.3 Uninstalling the Packages:**

**Important** Removing Debian package is allowed only when no other installed packages directly refer to it. For example removing `librealsense2-udev-rules` requires `librealsense2` to be removed first.

Remove a single package with:
  `sudo apt-get purge <package-name>`

Remove all RealSense™ SDK-related packages with:
  `dpkg -l | grep "realsense" | cut -d " " -f 3 | xargs sudo dpkg --purge`

#### 1.2.4 Realsense python package install

`pip install pyrealsense2`

## Aruco pose calibration
