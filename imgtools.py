from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from typing import Union, Tuple
from PIL import Image
import os 

import os
import io
import threading
from tkinter import *
from tkinter import filedialog, colorchooser, ttk
from PIL import Image, ImageTk
from rembg import remove

class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Background Remover")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.images = []
        self.output_format = StringVar(value="png")
        self.output_folder = StringVar()
        self.bg_color = "#FFFFFF"
        self.quality = IntVar(value=90)
        self.processed_images = []

        self.create_ui()

    def create_ui(self):
        # Sidebar
        sidebar = Frame(self.root, width=300, bg="#e4e4e4", padx=10, pady=10)
        sidebar.pack(side=LEFT, fill=Y)

        Button(sidebar, text="Select Images", command=self.select_images).pack(fill=X, pady=5)

        Label(sidebar, text="Output Format:", bg="#e4e4e4").pack(anchor=W, pady=(10,0))
        format_menu = ttk.Combobox(sidebar, textvariable=self.output_format, values=["png", "jpg", "webp"], state="readonly")
        format_menu.pack(fill=X)

        Button(sidebar, text="Choose Output Folder", command=self.select_output_folder).pack(fill=X, pady=5)

        Button(sidebar, text="Pick Background Color", command=self.pick_bg_color).pack(fill=X, pady=5)

        Label(sidebar, text="Quality (JPG/WebP):", bg="#e4e4e4").pack(anchor=W, pady=(10,0))
        Scale(sidebar, from_=1, to=100, orient=HORIZONTAL, variable=self.quality).pack(fill=X)

        Button(sidebar, text="Start Processing", bg="#4CAF50", fg="white", command=self.start_processing).pack(fill=X, pady=10)

        # Work area
        self.work_area = Canvas(self.root, bg="#ffffff")
        self.work_area.pack(side=RIGHT, expand=True, fill=BOTH)

    def select_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp")])
        if file_paths:
            self.images = list(file_paths)
            self.show_previews()

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)

    def pick_bg_color(self):
        color = colorchooser.askcolor(title="Choose Background Color")
        if color[1]:
            self.bg_color = color[1]

    def show_previews(self):
        self.work_area.delete("all")
        preview_size = 150
        cols = 4
        for i, img_path in enumerate(self.images[:8]):
            img = Image.open(img_path)
            img.thumbnail((preview_size, preview_size))
            photo = ImageTk.PhotoImage(img)
            row = i // cols
            col = i % cols
            x = col * (preview_size + 10) + 50
            y = row * (preview_size + 40) + 50
            self.work_area.create_image(x, y, image=photo, anchor=NW)
            self.work_area.image = photo

    def start_processing(self):
        if not self.images:
            return
        threading.Thread(target=self.process_images).start()

    def process_images(self):
        self.processed_images.clear()
        for img_path in self.images:
            with open(img_path, "rb") as f:
                raw = f.read()

            output = remove(raw)
            img = Image.open(io.BytesIO(output)).convert("RGBA")

            # Handle non-transparent formats
            if self.output_format.get() == "jpg":
                bg = Image.new("RGB", img.size, self.bg_color)
                bg.paste(img, mask=img.split()[3])
                img = bg

            elif self.output_format.get() == "webp":
                img = img.convert("RGBA")

            # Save processed file
            filename = os.path.splitext(os.path.basename(img_path))[0] + "." + self.output_format.get()
            save_path = os.path.join(self.output_folder.get() or os.path.dirname(img_path), filename)

            img.save(save_path, quality=self.quality.get())
            self.processed_images.append(save_path)

        self.show_after_previews()

    def show_after_previews(self):
        self.work_area.delete("all")
        preview_size = 150
        cols = 4
        for i, img_path in enumerate(self.processed_images[:8]):
            img = Image.open(img_path)
            img.thumbnail((preview_size, preview_size))
            photo = ImageTk.PhotoImage(img)
            row = i // cols
            col = i % cols
            x = col * (preview_size + 10) + 50
            y = row * (preview_size + 40) + 300
            self.work_area.create_image(x, y, image=photo, anchor=NW)
            self.work_area.image = photo



#This function takes a PIL image and returns the img in RGBA format 

def remove_bg(img:Image) -> Image:
    image_bg    = process(img)


DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IN_SIZE     = 1024

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE).float()

transform_image = transforms.Compose(
    [
        transforms.Resize((IN_SIZE, IN_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
#Error while processing rearrange-reduction pattern "b c (hg h) (wg w) -> b (c hg wg) h w". Input tensor shape: torch.Size([1, 3, 2992, 2992]). Additional info: {'hg': 31, 'wg': 31}. Shape mismatch, can't divide axis of length 2992 in chunks of 31
def fn(image: Union[Image.Image, str]) -> Tuple[Image.Image, Image.Image]:
    """
    Remove the background from an image and return both the transparent version and the original.

    This function performs background removal using a BiRefNet segmentation model. It is intended for use
    with image input (either uploaded or from a URL). The function returns a transparent PNG version of the image
    with the background removed, along with the original RGB version for comparison.

    Args:
        image (PIL.Image or str): The input image, either as a PIL object or a filepath/URL string.

    Returns:
        tuple:
            - processed_image (PIL.Image): The input image with the background removed and transparency applied.
            - origin (PIL.Image): The original RGB image, unchanged.
    """
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    processed_image = process(im)
    return (processed_image, origin)

def resize(image: Image.Image,max_dim:int=2048):

    image_size  = image.size 
    cr          = image_size[1] / image_size[0]

    if (image_size[0] > max_dim) or (image_size[1] > max_dim):

        if image_size[1] > image_size[0]:
            d1 = max_dim
            d2 = int(max_dim // cr)
        else:
            d1 = int(max_dim // cr)
            d2 = max_dim
        return (d1,d2)
    else:
        return image_size


@spaces.GPU
def process(image: Image.Image) -> Image.Image:
    """
    Apply BiRefNet-based image segmentation to remove the background.

    This function preprocesses the input image, runs it through a BiRefNet segmentation model to obtain a mask,
    and applies the mask as an alpha (transparency) channel to the original image.

    Args:
        image (PIL.Image): The input RGB image.

    Returns:
        PIL.Image: The image with the background removed, using the segmentation mask as transparency.
    """
    image_size  = resize(image)
    image       = image.resize(image_size)
    

      
    rotate = image_size[0] == image_size[1]
    if rotate:
        image = image.rotate(270)
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)

    
    return image

def process_file(f: str) -> str:
    """
    Load an image file from disk, remove the background, and save the output as a transparent PNG.

    Args:
        f (str): Filepath of the image to process.

    Returns:
        str: Path to the saved PNG image with background removed.
    """
    oldname         = f.split('.')[-1]
    newpath         = f.replace("."+oldname,'_no_bg.png')
    print(f"{f} -> {newpath}")
    im:Image.Image  = load_img(f, output_type="pil")
    im = im.convert("RGB")
    #if im.size[0] ==
    transparent = process(im)
    transparent.save(newpath)
    return None


def resize_folder(froot:str,same_level=False,new_dir='no_bg',ftype='jpeg'):

    for file in os.listdir(froot):
        load_path   = os.path.join(froot,file)
        if not os.path.isfile(load_path):
            continue
        if same_level:
            save_dir = os.path.join(os.path.dirname(froot),new_dir)
        else:
            save_dir    = os.path.join(froot,new_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path   = os.path.join(save_dir,file)
        if os.path.exists(save_path):
            continue 
        try:
            image       = load_img(load_path, output_type="pil").convert("RGB")
        except ValueError:
            print(f"failed to load img: {file}")
            continue
        image_bg    = process(image)#.rotate(270)

        if not ftype == 'png':
            #Convert to jpeg 
            white_bg_img    = Image.new('RGB',image_bg.size,color='#222829')
            white_bg_img.paste(image_bg,mask=image_bg.split()[3])
            white_bg_img.save(save_path,format='jpeg',quality=75)
        else:
            image_bg.save(save_path,format=ftype)
        #image_bg.convert("RGB").save(save_path,format='jpeg')
    
    print(f"complete")

def resize_folder2(froot:str):
    for file in os.listdir(froot):
        load_path   = os.path.join(froot,file)
        if not os.path.isfile(load_path):
            continue

        image:Image.Image       = load_img(load_path, output_type="pil")
        dims        = resize(image)
        image = image.resize(dims)
        try:
            #white_bg_img    = Image.new('RGB',image.size,(255,255,255))
            #white_bg_img.paste(image,mask=image.split()[3])
            image.save(load_path,format='jpeg',quality=60)
        except OSError as err:
            print(f"err on {file} - {err}")

            white_bg_img    = Image.new('RGB',image.size,color='#222829')
            white_bg_img.paste(image,mask=image.split()[3])
            white_bg_img.save(load_path,format='jpeg',quality=80)
            pass



if __name__ == "__main__":

    root = Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()


    exit()
    root    = "D:/#hipowerpc/PCs/FrostByte Nova/no_bg"
    root    = "C:/users/evere/Pictures/wheel"
    resize_folder(root,same_level=True,new_dir="no_bg_png",ftype='png')
    #resize_folder2("D:/HiPowerPC/media/site media/")
    #resize_folder2(root)
    #Gaming PC RGB FrostByte Nano - Budget Gaming PC