import wandb

def wandb_log(prefix, sp_values, com_values):
    """
        prefix + tags.values is the name of sp_values;
        values should include information like:
        {"Acc": 0.9, "Loss":}
        com_values should include information like:
        {"epoch": epoch, }
    """
    new_values = {}
    for k, _ in sp_values.items():
        new_values[prefix+"/" + k] = sp_values[k]
    new_values.update(com_values)
    wandb.log(new_values)



