import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm import tqdm 

def read_char_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_font_files(font_folder):
    return [os.path.join(font_folder, f)
            for f in os.listdir(font_folder)
            if f.lower().endswith('.ttf')]

def is_char_supported(font_path, char):
    try:
        font = TTFont(font_path)
        for table in font['cmap'].tables:  # character to glyph index mapping 
            if ord(char) in table.cmap:
                return True
    except Exception:
        pass
    return False

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
        print(f"Error rendering '{char}' with font {font_path}: {e}")
        return None

# check if a character is safe to use in a filename
def safe_filename(char): 
    if char in r'\/:*?"<>|':
        return f'uni{ord(char)}'
    return char

def generate_images(char_list_file, font_folder, output_folder,
                    img_size=(128, 128), font_size=100, mode='L'):
    os.makedirs(output_folder, exist_ok=True)
    all_chars = read_char_list(char_list_file)
    font_paths = get_font_files(font_folder)
    print(f"{len(font_paths)} fonts found in '{font_folder}'.")

    # check which characters are supported by all fonts
    supported_chars = []
    for char in all_chars:
        char_supported_by_all = True
        for font_path in font_paths:
            if not is_char_supported(font_path, char):
                print(f"Char '{char}' is not supported by font {os.path.basename(font_path)}")
                char_supported_by_all = False
                break
        if char_supported_by_all:
            supported_chars.append(char)
    print(f"{len(supported_chars)} / {len(all_chars)} characters are supported by all fonts.")

    for font_path in tqdm(font_paths, desc="Generating images", unit="font"):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        font_output_dir = os.path.join(output_folder, font_name)
        os.makedirs(font_output_dir, exist_ok=True)

        for char in supported_chars:
            img = render_char_image(char, font_path, img_size, font_size, mode)
            if img:
                filename = f"{font_name}_{safe_filename(char)}.png"
                img.save(os.path.join(font_output_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate character images from fonts.")
    parser.add_argument('--char_list', type=str, default='./char_list.txt', help='Path to character list')
    parser.add_argument('--fonts', type=str, default='./fonts', help='Font folder path')
    parser.add_argument('--output', type=str, default='./data', help='Output image folder')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], help='Image size (width height)')
    parser.add_argument('--font_size', type=int, default=100, help='Font size')
    parser.add_argument('--mode', type=str, choices=['L', 'RGB'], default='L', help='Image mode')

    args = parser.parse_args()

    generate_images(
        char_list_file=args.char_list,
        font_folder=args.fonts,
        output_folder=args.output,
        img_size=tuple(args.img_size),
        font_size=args.font_size,
        mode=args.mode
    )