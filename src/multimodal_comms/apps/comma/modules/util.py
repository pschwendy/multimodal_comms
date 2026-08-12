import functools
import tkinter as tk
from sys import platform

if platform in ('win32', 'darwin'):
    import win32gui
    import win32ui
    from ctypes import windll
    
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageGrab

def get_screenshot_of_puzzle_window(window_name):

    if platform not in ('win32', 'darwin'):
        return ImageGrab.grab()

    hwnd = win32gui.FindWindow(None, window_name)

    # Uncomment the following line if you use a high DPI display or >100% scaling size
    # windll.user32.SetProcessDPIAware()

    # Change the line below depending on whether you want the whole window
    # or just the client area. 
    #left, top, right, bot = win32gui.GetClientRect(hwnd)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # Change the line below depending on whether you want the whole window
    # or just the client area. 
    #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        return im
    

class HighlightButton(tk.Button):
    def __init__(self, module, x, y, width, height, font=('Helvetica', 16), **kwargs) -> None:
        super().__init__(module.game_manager.master, font = font, **kwargs)
        self.canvas = module.canvas
        self.module = module
        self.place(x=x, y=y, width=width, height=height)
        self.width = width
        self.height = height
        self.current_highlight_id = None
        self.bind("<Enter>", functools.partial(self.on_enter, x, y))
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, top_left_x, top_left_y, event=None):
        rect_id = self.canvas.create_rectangle(top_left_x - 2, top_left_y - 2, top_left_x + self.width, top_left_y + self.height, outline="orange", width=4)
        self.current_highlight_id = rect_id


    def on_leave(self, event):
        if self.current_highlight_id is not None:
            self.canvas.delete(self.current_highlight_id)
            self.current_highlight_id = None

    def save_to_image(self):
        FONTSIZE = 14

        # Create an off-screen image
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, self.width, self.height], fill="lightgray", outline="black")

        # Get button properties
        text = self['text']
        img = None
        if self['image']:
            img = self.image
            pil_image = ImageTk.getimage(img)

        if img:
            image.paste(pil_image, (0, 0, pil_image.width, pil_image.height))

        elif text:
        
            font = ImageFont.truetype("modules/ARIAL.TTF", FONTSIZE)  # Use a font available on your system
            text_width = draw.textlength(text, font=font)
            text_height = FONTSIZE
            text_x = (self.width - text_width) / 2
            text_y = (self.height - text_height) / 2

            # Draw the button text
            draw.text((text_x, text_y), text, fill="black", font=font)

        image.save("button.png")

        return image