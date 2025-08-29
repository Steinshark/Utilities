import gradio as gr
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from typing import Union, Tuple
from PIL import Image
import os 

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")

IN_SIZE     = 1024
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
    input_images = transform_image(image).unsqueeze(0).to("cuda")
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

# slider1 = gr.ImageSlider(label="Processed Image", type="pil", format="png")
# slider2 = gr.ImageSlider(label="Processed Image from URL", type="pil", format="png")
# image_upload = gr.Image(label="Upload an image")
# image_file_upload = gr.Image(label="Upload an image", type="filepath")
# url_input = gr.Textbox(label="Paste an image URL")
# output_file = gr.File(label="Output PNG File")

# # Example images
# chameleon = load_img("butterfly.jpg", output_type="pil")
# url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

# tab1 = gr.Interface(fn, inputs=image_upload, outputs=slider1, examples=[chameleon], api_name="image")
# tab2 = gr.Interface(fn, inputs=url_input, outputs=slider2, examples=[url_example], api_name="text")
# tab3 = gr.Interface(process_file, inputs=image_file_upload, outputs=output_file, examples=["butterfly.jpg"], api_name="png")

# demo = gr.TabbedInterface(
#     [tab1, tab2, tab3], ["Image Upload", "URL Input", "File Output"], title="Background Removal Tool"
# )

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
    root    = "D:/hipowerpc/PCs/FrostByte Nova/no_bg"
    #resize_folder(root,same_level=True,new_dir="no_bg_png",ftype='png')
    resize_folder2("D:/HiPowerPC/media/site media/")
    #resize_folder2(root)
    #Gaming PC RGB FrostByte Nano - Budget Gaming PC