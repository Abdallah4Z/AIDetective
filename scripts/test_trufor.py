import sys
import os
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add TruFor paths to system path
TRUFOR_PATH = Path(__file__).parent.parent / "src/models/pretrained/TruFor/test_docker/src"
sys.path.insert(0, str(TRUFOR_PATH.parent))
sys.path.insert(0, str(TRUFOR_PATH))

try:
    from config import update_config, _C as config
    from data_core import myDataset
except ImportError as e:
    print(f"Error importing TruFor modules: {e}")
    print(f"Please ensure TruFor is properly installed in: {TRUFOR_PATH}")
    sys.exit(1)


def initialize_config(model_weights=None):
    # Create a minimal args object for update_config
    class Args:
        opts = None
    
    args = Args()
    
    # Change to the directory containing trufor.yaml to ensure it's found
    original_dir = os.getcwd()
    config_dir = TRUFOR_PATH
    os.chdir(config_dir)
    
    try:
        # This will load trufor.yaml and populate all config fields
        update_config(config, args)
        
        # Override model file if provided
        if model_weights:
            config.defrost()
            config.TEST.MODEL_FILE = str(model_weights)
            config.freeze()
    finally:
        os.chdir(original_dir)


class TruForTester:
    
    def __init__(self, model_weights=None, device='cuda', save_npp=False, max_size=1024):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.save_npp = save_npp
        self.max_size = max_size
        
        # Default weights path
        if model_weights is None:
            model_weights = TRUFOR_PATH.parent / "weights/trufor.pth.tar"
        
        self.model_weights = Path(model_weights)
        
        if not self.model_weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_weights}\n"
                f"Please download from: https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
            )
        
        # Initialize config by loading trufor.yaml
        initialize_config(self.model_weights)
        
        # Update device-specific settings
        config.defrost()
        config.CUDNN.BENCHMARK = True if self.device == 'cuda' else False
        config.CUDNN.DETERMINISTIC = False
        config.CUDNN.ENABLED = True if self.device == 'cuda' else False
        config.freeze()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the TruFor model"""
        print(f"Loading model from: {self.model_weights}")
        print(f"Using device: {self.device}")
        
        checkpoint = torch.load(self.model_weights, map_location=torch.device(self.device), weights_only=False)

        
        if config.MODEL.NAME == 'detconfcmx':
            from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
            self.model = confcmx(cfg=config)
        else:
            raise NotImplementedError(f"Model {config.MODEL.NAME} not implemented")
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Clear GPU cache after loading model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        print("Model loaded successfully!")
    
    def process_image(self, image_path):
        # Load image
        img_PIL = Image.open(image_path).convert("RGB")
        original_size = img_PIL.size  # (width, height)
        
        # Resize if image is too large
        if self.max_size is not None:
            max_dim = max(original_size)
            if max_dim > self.max_size:
                scale = self.max_size / max_dim
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                img_PIL = img_PIL.resize(new_size, Image.LANCZOS)
                print(f"Resized image from {original_size} to {new_size} for memory efficiency")
        
        img_RGB = np.array(img_PIL)
        rgb_tensor = torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            pred, conf, det, npp = self.model(rgb_tensor)
            
            # Process outputs
            results = {}
            results['imgsize'] = tuple(rgb_tensor.shape[2:])
            
            # Localization map
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            results['map'] = pred.cpu().numpy()
            
            # Detection score
            if det is not None:
                results['score'] = torch.sigmoid(det).item()
            
            # Confidence map
            if conf is not None:
                conf = torch.squeeze(conf, 0)
                conf = torch.sigmoid(conf)[0]
                results['conf'] = conf.cpu().numpy()
            
            # Noiseprint++
            if npp is not None and self.save_npp:
                npp = torch.squeeze(npp, 0)[0]
                results['np++'] = npp.cpu().numpy()
        
        # Store original image size for reference
        results['original_size'] = original_size
        
        # Clear GPU cache to prevent memory accumulation
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return results
    
    def process_batch(self, input_path, output_dir, overwrite=False):
        # Get list of images
        if '*' in input_path:
            image_list = glob(input_path, recursive=True)
            image_list = [img for img in image_list if not os.path.isdir(img)]
        elif os.path.isfile(input_path):
            image_list = [input_path]
        elif os.path.isdir(input_path):
            image_list = glob(os.path.join(input_path, '**/*'), recursive=True)
            image_list = [img for img in image_list if not os.path.isdir(img)]
            # Filter for image files
            image_list = [img for img in image_list if img.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        if not image_list:
            print("No images found!")
            return
        
        print(f"Found {len(image_list)} images to process")
        
        # Process each image
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_summary = []
        
        for img_path in tqdm(image_list, desc="Processing images"):
            try:
                # Determine output path
                img_name = Path(img_path).stem
                output_file = output_dir / f"{img_name}.npz"
                
                # Skip if exists and not overwriting
                if output_file.exists() and not overwrite:
                    print(f"Skipping {img_path} (output exists)")
                    continue
                
                # Process image
                results = self.process_image(img_path)
                
                # Save results
                np.savez(output_file, **results)
                
                # Store summary
                summary = {
                    'image': img_path,
                    'score': results.get('score', None),
                    'output': str(output_file)
                }
                results_summary.append(summary)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return results_summary
    
    def visualize_results(self, image_path, results, mask_path=None, save_path=None):
        # Load results if path provided
        if isinstance(results, (str, Path)):
            results = dict(np.load(results))
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) / 255.0
        
        # Create visualization
        n_plots = 3 if mask is None else 4
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Detection map
        det_map = results['map']
        im1 = axes[1].imshow(det_map, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"Detection Map\nScore: {results.get('score', 'N/A'):.4f}")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Confidence map
        if 'conf' in results:
            conf_map = results['conf']
            im2 = axes[2].imshow(conf_map, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title("Confidence Map")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Ground truth mask if provided
        if mask is not None:
            axes[3].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title("Ground Truth Mask")
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test TruFor model for image forgery detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single image
    python test_trufor.py -i data/test_image.jpg -o results/
    
    # Test all images in directory
    python test_trufor.py -i data/images/ -o results/
    
    # Test with glob pattern
    python test_trufor.py -i "data/**/*.jpg" -o results/
    
    # Visualize results
    python test_trufor.py -i data/test_image.jpg -o results/ --visualize
    
    # Use CPU
    python test_trufor.py -i data/test_image.jpg -o results/ --device cpu
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input: single image, directory, or glob pattern')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('-w', '--weights', type=str, default=None,
                        help='Path to model weights (.pth.tar)')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--max_size', type=int, default=1024,
                        help='Maximum image dimension (default: 1024). Set to 0 to disable resizing.')
    parser.add_argument('--save_np', action='store_true',
                        help='Save Noiseprint++ output')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualizations for results (default: True)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='Disable visualization generation')
    parser.add_argument('--mask', type=str, default=None,
                        help='Ground truth mask path (for single image visualization)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Initialize tester
    try:
        max_size = args.max_size if args.max_size > 0 else None
        tester = TruForTester(
            model_weights=args.weights,
            device=args.device,
            save_npp=args.save_np,
            max_size=max_size
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Process images
    print(f"\nProcessing images from: {args.input}")
    print(f"Output directory: {args.output}\n")
    
    results_summary = tester.process_batch(
        args.input,
        args.output,
        overwrite=args.overwrite
    )
    
    # Print summary
    if results_summary:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for result in results_summary:
            score = result.get('score')
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"{Path(result['image']).name}: Score = {score_str}")
        print("="*60)
    
    # Visualize if requested
    if args.visualize and results_summary:
        print("\nGenerating visualizations...")
        viz_dir = Path(args.output) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for result in results_summary:
            img_name = Path(result['image']).stem
            viz_path = viz_dir / f"{img_name}_visualization.png"
            
            tester.visualize_results(
                result['image'],
                result['output'],
                mask_path=args.mask,
                save_path=viz_path
            )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
