from matplotlib import pyplot as plt
from PIL import Image
import os

if __name__ == '__main__':
    img_path = '/data/dataset/PV/003-XinXie/images/DJI_20251124150621_0152_V.JPG'
    prompt_dir = '/data/dataset/PV/ZS_PV/003-XinXie/iter_0/views/view_5/prompts_mv_voting/'
    img = Image.open(img_path)
    
    plt.imshow(img)
    plt.axis('off')
    prompt_file = os.path.join(prompt_dir, 'DJI_20251124150621_0152_V.txt')
    with open(prompt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            x1, y1, x2, y2 = parts[:4]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, edgecolor='red', facecolor='none', linewidth=0.25)
            plt.gca().add_patch(rect)
    plt.savefig('../tmp/iter_0_DJI_20251124150621_0152_V_mv_voting.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    
    