import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import json
from tqdm import tqdm
from dreamsim import dreamsim
import hashlib

from adain_model import AdaINModel

device = "cuda" if torch.cuda.is_available() else "cpu"
vgg_path = "models/vgg_normalised.pth"
decoder_path = "models/decoder.pth"
stylized_dir  = "data/stylized"

class PairDataset(Dataset):
    def __init__(self, content_paths, style_paths, alphas, transform=None):
        self.content_paths = content_paths
        self.style_paths = style_paths
        self.alphas = alphas
        self.transform = transform or transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.content_paths) * len(self.style_paths) * len(self.alphas)

    def __getitem__(self, idx):
        n_c = len(self.content_paths)
        n_s = len(self.style_paths)
        n_a = len(self.alphas)

        i = idx // (n_s * n_a)
        j = (idx % (n_s * n_a)) // n_a
        k = idx % n_a

        content_path = self.content_paths[i]
        style_path = self.style_paths[j]
        alpha = self.alphas[k]

        content = Image.open(content_path).convert("RGB")
        style = Image.open(style_path).convert("RGB")

        if self.transform:
            content = self.transform(content)
            style = self.transform(style)

        return content, style, alpha, content_path, style_path

def load_image_list(root, exts=(".jpg", ".png")):
    paths = []
    for fname in sorted(os.listdir(root)):
        if fname.lower().endswith(exts):
            paths.append(os.path.join(root, fname))
    return paths

def hash_file_name(content_path, style_path, alpha_val):
    content_name = os.path.basename(content_path).split('.')[0]
    style_name = os.path.basename(style_path).split('.')[0]
    hash_id = hashlib.md5(f"{content_name}_{style_name}_{alpha_val}".encode()).hexdigest()[:8]
    return f"{hash_id}.jpg"

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="data/content")
    parser.add_argument("--style_dir", type=str, default="data/style")
    parser.add_argument("--output_json", type=str, default="data/cycle_scores.json")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    # Load model
    model = AdaINModel(vgg_path=vgg_path, decoder_path=decoder_path).to(device).eval()
    dreamsim_model, _ = dreamsim(pretrained=True, device=device)

    # Load data
    content_paths = load_image_list(args.content_dir)[:50]
    style_paths = load_image_list(args.style_dir)[:10]
    dataset = PairDataset(content_paths, style_paths, args.alphas)
    
    if len(dataset) > args.max_samples:
        print(f"Warning: dataset size {len(dataset)} > max_samples {args.max_samples}. Truncating.")
        # Simple subsample by slicing
        from torch.utils.data import Subset
        indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    results = []

    for batch in tqdm(dataloader, desc="Generating cycle scores"):
        content, style, alpha, content_path, style_path = batch
        content = content.to(device)
        style = style.to(device)

        # Forward AdaIN
        alpha_val = alpha.item()
        orig = content
        stylized, stats = model(orig, style, alpha=alpha_val)

        # Inverse cycle
        stylized_feat = model.encode(stylized)
        rec_feat = model.inverse_adain(stylized_feat, stats)
        rec = model.decode(rec_feat)

        # Compute DREAMSim
        dist = dreamsim_model(orig, rec).item()
        reward = 1.0 - dist

        # Save stylized image
        content_path_val = content_path[0]
        style_path_val = style_path[0]
        stylized_path = os.path.join(stylized_dir, hash_file_name(content_path_val, style_path_val, alpha_val))
        save_image(stylized.squeeze(0).cpu(), stylized_path)

        results.append({
            "content_path": content_path_val,
            "style_path": style_path_val,
            "alpha": alpha_val,
            "stylized_path": stylized_path,
            "reward": reward,
            "dreamsim_distance": dist
        })

    # Save
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} samples to {args.output_json}")

if __name__ == "__main__":
    main()