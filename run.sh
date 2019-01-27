#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

export PYTHONPATH=$SCRIPTPATH/src:$PYTHONPATH
python -m puddle.run
