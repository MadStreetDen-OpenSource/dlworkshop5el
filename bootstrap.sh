#!/usr/bin/env bash
apt-get install -y  python-dev
apt-get install -y python-numpy
apt-get install -y cython
apt-get install -y python-scipy
apt-get install -y python-matplotlib
apt-get update
if ! [ -L /var/www ]; then
  rm -rf /var/www
  ln -fs /vagrant /var/www
fi
