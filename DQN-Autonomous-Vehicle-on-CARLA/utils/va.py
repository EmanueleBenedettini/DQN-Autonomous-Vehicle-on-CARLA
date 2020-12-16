import cv2 as cv
import numpy as np
import IPython
import html
import base64
from ipywidgets import IntSlider


def show(*images, enlarge_small_images = True, max_per_row = -1, font_size = 0):
    """
    Visualizza una o più immagini nell'output della cella, 
    con un eventuale titolo sopra ciascuna di esse.
    Esempi di utilizzo:
    - va.show(img) : visualizza l'immagine img
    - va.show(img1, img2, img3) : visualizza le immagini img1, img2 e img3
    - va.show(img, title) : visualizza l'immagine img con il titolo title
    - va.show((img1, title1), (img2, title2)) : visualizza le immagini img1 e img2 con i rispettivi titoli title1 e title2
    """
    if len(images) == 2 and type(images[1])==str:
        # Gestisce il caso in cui è chiamato solo con immagine e titolo separatamente, senza essere una tupla
        images = [(images[0], images[1])]

    def convert(imgOrTuple):
        try:
            img, title = imgOrTuple
            if type(title)!=str: # "Falso positivo", assume fosse solo immagine senza titolo
                img, title = imgOrTuple, ''
        except ValueError: # Non può fare unpack: assume sia solo un'immagine senza titolo
            img, title = imgOrTuple, ''        
        if type(img)==str:
            data = img    # Suppone sia il path
        else:
            img = convert_for_display(img)
            if enlarge_small_images:
                REF_SCALE = 400
                h, w = img.shape[:2]
                if h<REF_SCALE or w<REF_SCALE:
                    # Immagini molto piccole vengono ingrandite
                    scale = max(1, min(REF_SCALE//h, REF_SCALE//w))
                    img = cv.resize(img,(w*scale,h*scale), interpolation=cv.INTER_NEAREST)
            data = 'data:image/png;base64,' + base64.b64encode(cv.imencode('.png', img)[1]).decode('utf8')
        return data, title
    
    if max_per_row == -1:
        max_per_row = len(images)
    
    rows = [images[x:x+max_per_row] for x in range(0, len(images), max_per_row)]
    font = f"font-size: {font_size}px;" if font_size else ""
    
    html_content = ""
    for r in rows:
        l = [convert(t) for t in r]
        html_content += "".join(["<table><tr>"] 
                + [f"<td style='text-align:center;{font}'>{html.escape(t)}</td>" for _,t in l]    
                + ["</tr><tr>"] 
                + [f"<td style='text-align:center;'><img src='{d}'></td>" for d,_ in l]
                + ["</tr></table>"])
    IPython.display.display(IPython.display.HTML(html_content))

def convert_for_display(img):
    if img.dtype!=np.uint8:
        # Valori dei pixel non byte
        a, b = img.min(), img.max()
        if a==b:
            offset, mult, d = 0, 0, 1
        elif a<0:
            # Ci sono dei valori negativi: riscala facendo corrispondere lo 0 a 128
            offset, mult, d = 128, 127, max(abs(a), abs(b))
        else:
            # Tutti valori positivi o zero: riscala [0,max] in [0,255]
            offset, mult, d = 0, 255, b
        # Normalizza e trasforma in byte
        img = np.clip(offset + mult*(img.astype(float))/d, 0, 255).astype(np.uint8)
    return img
        
def center_text(img, text, center, color, fontFace = cv.FONT_HERSHEY_PLAIN, fontScale = 1, thickness = 1, lineType = cv.LINE_AA, max_w = -1):
    """
    Utilizza cv.getTextSize e cv.putText per centrare il testo nel punto center.
    """
    while True:
        (w, h), _ = cv.getTextSize(text, fontFace, fontScale, thickness)
        if max_w<0 or w<max_w or fontScale<0.2:
            break
        fontScale *= 0.8
    pt = (center[0]-w//2, center[1]+h//2)
    cv.putText(img, text, pt, fontFace, fontScale, color, thickness, lineType)

    
def draw_hist(hist, height = 192, back_color = (160,225,240), border = 5):
    """
    Restituisce un'immagine con il disegno dell'istogramma.
    """
    size = hist.size
    img = np.full((height, size+border*2, 3), back_color, dtype=np.uint8)
    nh = np.empty_like(hist, dtype=np.int32)
    cv.normalize(hist, nh, 0, height-1-border*2, cv.NORM_MINMAX, cv.CV_32S)
    for i in range(size):
        img[-border-nh[i]:-border,border+i,0:3] = i
    return img    
