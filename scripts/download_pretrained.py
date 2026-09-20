"""Download Depth Anything V2 checkpoints (vitb / vits) from Hugging Face."""

import argparse
import os
import urllib.request

URLS = {
    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', type=str, default='vitb', choices=list(URLS))
    p.add_argument('--outdir', type=str, default='pretrained')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    url = URLS[args.encoder]
    out = os.path.join(args.outdir, f'depth_anything_v2_{args.encoder}.pth')
    print(f'downloading {url}')
    urllib.request.urlretrieve(url, out)
    print(f'saved to {out}')


if __name__ == '__main__':
    main()
