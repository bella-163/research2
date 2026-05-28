import timm

patterns = [
    "*vit_base_patch16_224*in21k*",
    "*vit_small_patch16_224*in21k*",
    "*vit_tiny_patch16_224*in21k*",
]

for pattern in patterns:
    print(f"\n=== {pattern} ===")
    models = timm.list_models(pattern, pretrained=True)
    if not models:
        print("No pretrained models found for this pattern.")
    for name in models:
        print(name)
