#! /bin/bash

type=$1
##BASE_PATH = "/home/ec2-user/eta_training/eta_train/eta_venv/bin/activate"
`source /home/ec2-user/eta_training/eta_train/eta_venv/bin/activate && python /home/ec2-user/eta_training/eta_train/gen_count.py -t $1`
