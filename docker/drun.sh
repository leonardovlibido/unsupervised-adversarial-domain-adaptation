docker run \
    -it \
    --rm \
    --gpus device=6 \
    --shm-size 50G \
    --name evm-rdj-gpu1 \
    -v /home/rdjordjevic/master/repos/domain-adaptation-codebase/:/home/rdjordjevic/master/repos/domain-adaptation-codebase/ \
    domain-adaptation:latest

#-v /mnt/bgnas02-cold/Evermobile/:/mnt/bgnas02-cold/Evermobile/ \

