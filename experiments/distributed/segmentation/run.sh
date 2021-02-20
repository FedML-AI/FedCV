{
 ./run_fedseg_distributed_pytorch.sh 4 4 0 4 deeplabV3_plus resnet True 16 hetero 500 2 10 sgd 0.001 pascal_voc "/home/chaoyanghe/BruteForce/FedML/data/pascal_voc/benchmark_RELEASE" 0 > deeplab_resnet_hetero_ft_b10_r200e2l001 2>&1
}&
wait
kill -- -$$
exit 0

