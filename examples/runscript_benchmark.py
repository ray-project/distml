import os


model_name = "--model-name {}"
models = ["resnet18", "resnet101"]

for model in models:
    cmd = f"python benchmark_mnist_jax.py {model_name.format(model)}"
    print("Running command", cmd)
    os.system(cmd)

cmd = f"python benchmark_wiki_flax.py"
print("Running command", cmd)
os.system(cmd)

