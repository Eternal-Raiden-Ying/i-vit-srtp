import os


target_dir = "/root/autodl-tmp/imagenet/val"  # val_dir
file_dir = "/root/autodl-tmp/i-vit-srtp/dataset_prepare"

if __name__ == "__main__":

    assert os.path.exists(target_dir), "specify the directory of validate"
    assert os.path.exists(file_dir), "specify the directory of file, where the *.txt is"

    os.chdir(file_dir)
    assert os.path.exists("mkdir.txt")
    assert os.path.exists("categories.txt")


    os.chdir(target_dir)

    with open(os.path.join(file_dir, "mkdir.txt"), 'r') as cmdline:
        for line in cmdline:
            os.system(line)

    with open(os.path.join(file_dir, "categories.txt"), 'r') as cmdline:
        for line in cmdline:
            os.system(line)