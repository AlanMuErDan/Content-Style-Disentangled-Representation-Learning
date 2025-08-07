# font_gen.py

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm import tqdm 
import io



def read_char_list(file_path): # read a list of characters 
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]



def get_font_files(font_folder): # return a list of ttf files paths 
    return [os.path.join(font_folder, f)
            for f in os.listdir(font_folder)
            if f.lower().endswith('.ttf')]



def extract_supported_chars(font_path): # return a set of supported characters in integers (Unicode codepoints) 
    try:
        font = TTFont(font_path)
        supported = set()
        for table in font['cmap'].tables:
            supported.update(table.cmap.keys())
        return supported
    except Exception:
        return set()



def render_char_image(char, font_path, img_size, font_size, mode):
    try:
        img = Image.new(mode, img_size, color=255 if mode == 'L' else (255, 255, 255)) # create a white background
        draw = ImageDraw.Draw(img) 
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1] # calculate text width and height
        x = (img_size[0] - text_w) / 2 - bbox[0] # center the text horizontally
        y = (img_size[1] - text_h) / 2 - bbox[1] # center the text vertically
        draw.text((x, y), char, fill=0 if mode == 'L' else (0, 0, 0), font=font)
        return img
    except Exception as e:
        # print(f"Error rendering '{char}' with font {font_path}: {e}")
        return None



def safe_filename(char): 
    if char in r'\/:*?"<>|':
        return f'uni{ord(char)}'
    return char



def generate_images(char_list_file, font_folder, output_folder,
                    img_size=(128, 128), font_size=100, mode='L',
                    use_lmdb=False, lmdb_path=None):

    all_chars = [c for c in read_char_list(char_list_file) if len(c) == 1]
    font_paths = get_font_files(font_folder)
    print(f"{len(font_paths)} fonts found in '{font_folder}'.")
    os.makedirs(output_folder, exist_ok=True)
    char_counts = {} # supported chatacter counting dict 

    if use_lmdb:
        import lmdb
        assert lmdb_path is not None, "lmdb_path must be specified when use_lmdb is True"
        env = lmdb.open(lmdb_path, map_size=1 << 40)  # 1 TB space
        txn = env.begin(write=True)

    for font_path in tqdm(font_paths, desc="Generating images", unit="font"):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        if not use_lmdb: 
            font_output_dir = os.path.join(output_folder, font_name)
            os.makedirs(font_output_dir, exist_ok=True)

        supported_codepoints = extract_supported_chars(font_path)
        supported_chars = [char for char in all_chars if ord(char) in supported_codepoints]

        char_counts[font_name] = len(supported_chars)
        print(f"{len(supported_chars)} / {len(all_chars)} characters supported by {font_name}")

        for char in tqdm(supported_chars, desc=f"Rendering {font_name}", unit="char"):
            img = render_char_image(char, font_path, img_size, font_size, mode)
            if img:
                if use_lmdb:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    key = f"{font_name}+{safe_filename(char)}".encode()
                    txn.put(key, buffer.getvalue())
                else: 
                    filename = f"{font_name}+{safe_filename(char)}.png"
                    img_path = os.path.join(font_output_dir, filename)
                    img.save(img_path)

    if use_lmdb:
        txn.commit()
        env.close()
        print(f"Saved images to LMDB at {lmdb_path}")

    # Plot statistics
    # try:
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     counts = list(char_counts.values())
    #     font_names = list(char_counts.keys())

    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(counts, bins=20, kde=True)
    #     plt.title("Distribution of Supported Characters per Font")
    #     plt.xlabel("Number of Supported Characters")
    #     plt.ylabel("Number of Fonts")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_folder, "font_char_support_distribution.png"))
    #     plt.close()
    #     print("Saved support distribution plot.")
    # except ImportError:
    #     print("matplotlib/seaborn not installed; skipping plot.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate character images from fonts.")
    parser.add_argument('--char_list', type=str, default='./char_list.txt', help='Path to character list')
    parser.add_argument('--fonts', type=str, default='./fonts', help='Font folder path')
    parser.add_argument('--output', type=str, default='./data', help='Output image folder')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], help='Image size (width height)')
    parser.add_argument('--font_size', type=int, default=100, help='Font size')
    parser.add_argument('--mode', type=str, choices=['L', 'RGB'], default='L', help='Image mode')
    parser.add_argument('--use_lmdb', action='store_true', help='Whether to also save images to LMDB')
    parser.add_argument('--lmdb_path', type=str, default='./font_data.lmdb', help='Path to LMDB output file')

    args = parser.parse_args()

    generate_images(
        char_list_file=args.char_list,
        font_folder=args.fonts,
        output_folder=args.output,
        img_size=tuple(args.img_size),
        font_size=args.font_size,
        mode=args.mode,
        use_lmdb=args.use_lmdb,
        lmdb_path=args.lmdb_path
    )