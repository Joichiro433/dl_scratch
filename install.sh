#!/bin/bash
cd `dirname $0`

poetry install

echo `pwd`/Deep_learning_from_scratch > $(python -c 'import sys; print(sys.path)' | grep -o "[^']*/site-packages")/module.pth

