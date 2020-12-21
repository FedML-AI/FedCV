
# PyTorch DDP classification

## lr_scheduler parameter reference:




```
# sh run_classification.sh 8 2 0 192.168.11.1 11111

```













```
# kill all processes
kill $(ps aux | grep "ddp_classification.py" | grep -v grep | awk '{print $2}')
```