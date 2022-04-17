def parse(process_id, worker_number, gpu_util_parse):
    gpu_util_parse_temp = gpu_util_parse.split(';')
    gpu_util_parse_temp = [(item.split(':')[0], item.split(':')[1]) for item in gpu_util_parse_temp ]

    gpu_util = {}
    for (host, gpus_str) in gpu_util_parse_temp:
        gpu_util[host] = [int(num_process_on_gpu) for num_process_on_gpu in gpus_str.split(',')]

    gpu_util_map = {}
    i = 0
    for host, gpus_util_map_host in gpu_util.items():
        for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
            for _ in range(num_process_on_gpu):
                gpu_util_map[i] = (host, gpu_j)
                i += 1
    print(gpu_util_map)


parse(0, 10, "local:2,2,0,6")






