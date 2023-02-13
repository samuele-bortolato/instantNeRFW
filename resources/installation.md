# Installation

The libraries used are constantly being updated and a month after the beginning of the project our code is already not fully compatible with the latest version, so we have to compile from source.
Following the instructions to install kaolin-wisp

### 1. Create an anaconda environment
```bash
conda create -n wisp python=3.9
conda activate wisp
pip install --upgrade pip
```
### 2. Clone repository
```bash
git clone git@github.com:NVIDIAGameWorks/kaolin-wisp.git
cd kaolin-wisp
git checkout 351a04a7daadbf57d4fd96e9f74ff77ffe809e19
```
### 3. Install PyTorch
note: not all pytorch versions are compatible with the libraries
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
### 4. Install OpenEXR
Install OpenEXR on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install libopenexr-dev 
```

Install OpenEXR on Windows:

```bash
pip install pipwin
pipwin install openexr
```
### 5. Install Kaolin
```bash
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout a50ed1324a3c84e1bee2af1087688461f4b98fe7
python setup.py develop
cd ..
```
### 5. Install requirements for the library
```bash
pip install -r requirements.txt
```
(Optional) for the GUI, highly recommended
```bash
pip install -r requirements_app.txt
```
other requirements that were not in the text
```bash
pip install PyEXR
pip install lpips
```
### 6. Install the library
```bash
python setup.py develop

```


