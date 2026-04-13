from pathlib import Path


TARGET = Path("SLAM/mega-sam/UniDepth/scripts/demo_mega-sam.py")
OLD = '''  print("Torch version:", torch.__version__)
  # model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
  # model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
  model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  demo(model, args)
'''
NEW = '''  print("Torch version:", torch.__version__, flush=True)
  model_path = os.environ.get("UNIDEPTH_MODEL_PATH", "lpiccinelli/unidepth-v2-vitl14")
  revision = os.environ.get("UNIDEPTH_MODEL_REVISION", "1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
  print(f"Loading UniDepthV2 from {model_path}", flush=True)
  if os.path.isdir(model_path):
    model = UniDepthV2.from_pretrained(model_path)
  else:
    model = UniDepthV2.from_pretrained(model_path, revision=revision)
  print("UniDepthV2 model loaded.", flush=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Moving UniDepthV2 model to {device}.", flush=True)
  model = model.to(device)
  print("UniDepthV2 model is ready for inference.", flush=True)
  demo(model, args)
'''


def main() -> None:
    text = TARGET.read_text()
    if NEW in text:
        print("Native UniDepth loader patch already present.")
        return
    if OLD not in text:
        raise RuntimeError(f"Could not find expected UniDepth loader block in {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"Patched native UniDepth loader in {TARGET}")


if __name__ == "__main__":
    main()
