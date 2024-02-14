"""Submit mutilple jobs."""
import threading, subprocess, argparse, glob, json, os
import concurrent.futures


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="-1")
parser.add_argument("--func", default="all")
args = parser.parse_args()

devices = args.gpu.split("/")
n_slots = len(devices)
functions = {}


def register(func):
    """Decorator to register a function in a dictionary."""
    global functions
    functions[func.__name__] = func
    return func


@register
def recon():  # deprecated
    cmd = "python train_simple.py -s ../cdmPanoHead/colmap_data/seed{idx} --model_path PanoHead_185_random/seed{idx} --white_background"
    for idx in range(100):
        yield cmd.format(idx=idx)

#python train_simple.py -s ../../data/objaverse-XL/renders_explicit/000081 -m ../../expr/GSurface/ --arch original  --random_init 1024 --white_background

# python train_simple.py -s ../../data/objaverse-XL/renders_explicit/000081 -m ../../expr/GSurface/ --arch original  --random_init 1024 --white_background


@register
def render_rotate():
    models = [
        'logs/2024-02-08T20-06-11_sd-face/checkpoints/last.ckpt',
        #'../../threestudio/load/zero123/stable_zero123.ckpt'
        ]
    model_name = [
        'trained_stablezero123',
        #'orig_stablezero123'
    ]
    cmd = "python inference.py --model {model} --name {name}"
    for model, name in zip(models, model_name):
        yield cmd.format(model=model, name=name)



def assign_slots(func_name):
    slot_cmds = [[] for _ in devices]
    for idx, cmd in enumerate(functions[func_name]()):
        device_id = idx % n_slots
        device_prefix = ""
        if devices[device_id] != -1:
            device_prefix = f"CUDA_VISIBLE_DEVICES={devices[device_id]} "
        slot_cmds[device_id].append(f"{device_prefix}{cmd}")
    return slot_cmds


def worker(device_id, device_cmds):
    for idx, cmd in enumerate(device_cmds):
        print(f"=> Device [{device_id}] GPU [{devices[device_id]}]: Starting task {idx} / {len(device_cmds)}")
        print(cmd)
        subprocess.run(cmd, shell=True)


slot_cmds = assign_slots(args.func)

with concurrent.futures.ThreadPoolExecutor(len(devices)) as executor:
    # Submit each command for execution
    futures = [executor.submit(worker, device_id, device_cmds) for device_id, device_cmds in enumerate(slot_cmds)]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)
