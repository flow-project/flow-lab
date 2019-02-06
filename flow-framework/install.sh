#!/usr/bin/env bash

# specify operating system
echo "Type the number corresponding to your operating system, followed by [ENTER]:"
echo "1 - Ubuntu 14.04"
echo "2 - Ubuntu 16.04"
echo "3 - Ubuntu 18.04"
echo "4 - Mac OSX"
read input_os

if (( "$input_os" != "1" )) && (( "$input_os" != "2" )) && (( "$input_os" != "3" )) && (( "$input_os" != "4" )); then
    echo "Value is not valid, exiting..."
    exit 1  # terminate and indicate error
fi

# move to the flow-framework directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}

# create conda environment
conda env create -f environment.yml
source activate flow-framework
ENV_PATH="$(dirname "$(which python)")"

# install dependencies
mkdir flow-framework && cd flow-framework

# -- rllab
git clone https://github.com/cathywu/rllab-multiagent
pushd rllab-multiagent
git checkout aboudy_patch
python setup.py develop
popd

# -- Flow
git clone https://github.com/flow-project/flow
pushd flow
git checkout f626cc5f030247471cd9ecdd32b6686c76d8ac3a
python setup.py develop
popd

# -- sumo
mkdir -p bin
pushd bin

if (( "$input_os" == "1" )); then  # Ubuntu 14.04
    # dependencies
    sudo apt-get update
    sudo apt-get install -y subversion autoconf build-essential libtool
    sudo apt-get install -y libxerces-c3.1 libxerces-c3-dev libproj-dev
    sudo apt-get install -y proj-bin proj-data libgdal1-dev libfox-1.6-0
    sudo apt-get install -y libfox-1.6-dev
    # binaries
    wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-ubuntu1404.tar.xz
    tar -xf binaries-ubuntu1404.tar.xz
    rm binaries-ubuntu1404.tar.xz
elif (( "$input_os" == "2" )); then  # Ubuntu 16.04
    # dependencies
    sudo apt-get update
    sudo apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
    sudo apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
    sudo apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
    sudo apt-get install -y build-essential curl unzip flex bison python python-dev
    pip install cmake cython
    # binaries
    wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-ubuntu1604.tar.xz
    tar -xf binaries-ubuntu1604.tar.xz
    rm binaries-ubuntu1604.tar.xz
elif (( "$input_os" == "3" )); then  # Ubuntu 18.04
    # dependencies
    sudo apt-get update
    sudo apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
    sudo apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
    sudo apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
    sudo apt-get install -y build-essential curl unzip flex bison python python-dev
    pip install cmake cython
    # binaries
    wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-ubuntu1804.tar.xz
    tar -xf binaries-ubuntu1804.tar.xz
    rm binaries-ubuntu1804.tar.xz
elif (( "$input_os" == "4" )); then  # Mac OSX
    # dependencies
    brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
    brew install Caskroom/cask/xquartz autoconf automake pkg-config libtool
    brew install gdal proj xerces-c fox
    # binaries
    wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-mac.tar.xz
    tar -xf binaries-mac.tar.xz
    rm binaries-mac.tar.xz
fi

chmod +x *
mv * ${ENV_PATH}
popd
rm -r bin/

if (( "$input_os" != "4" )); then
    echo "export SUMO_HOME="${SUMO_HOME}":"${ENV_PATH}"" >> ~/.bashrc
else
    echo "export SUMO_HOME="${SUMO_HOME}":"${ENV_PATH}"" >> ~/.bash_profile
fi
