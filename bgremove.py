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
        self.root.configure(bg="#2b2b2b")

        # Data
        self.images = []  # all selected originals
        self.processed_images = []  # saved outputs
        self.output_format = StringVar(value="png")
        self.output_folder = StringVar()
        self.bg_color = "#FFFFFF"
        self.quality = IntVar(value=90)
        self.image_rotations = {}  # path -> angle
        self.photo_refs = []  # keep PhotoImage refs alive
        self.before_shown = set()  # track which originals are shown
        self.after_shown = set()   # track which processed are shown

        self.create_ui()

    # ---------------- UI ----------------
    def create_ui(self):
        # Sidebar
        SIDEBAR_WIDTH   = 350
        sidebar = Frame(self.root, width=350, bg="#3a3a3a", padx=10, pady=10)
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)
        # --- Logo Section ---
        logo_frame = Frame(sidebar, bg="#3a3a3a")
        logo_frame.pack(fill=X, pady=(5, 15))  # Slight top margin for spacing

        try:
            # Load & resize logo
            IMG_WIDTH = SIDEBAR_WIDTH - 40
            logo_img = Image.open(r"C:\Users\evere\Pictures\logo.png")
            w, h = logo_img.size
            aspect_ratio = w / h
            new_height = int(IMG_WIDTH / aspect_ratio)

            # Resize cleanly
            logo_img = logo_img.resize((IMG_WIDTH, new_height), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)

            # Place logo image
            logo_label = Label(logo_frame, image=self.logo_photo, bg="#3a3a3a")
            logo_label.pack(anchor="center", pady=5)

        except Exception as e:
            # Fallback if logo fails to load
            fallback_label = Label(
                logo_frame,
                text="Background Remover",
                font=("Segoe UI", 18, "bold"),
                fg="white",
                bg="#3a3a3a"
            )
            fallback_label.pack(anchor="center", pady=10)
            print("Logo failed to load:", e)

        Button(sidebar, text="Select Images", command=self.select_images).pack(fill=X, pady=5)

        Label(sidebar, text="Output Format:", bg="#3a3a3a", fg="white").pack(anchor=W, pady=(10,0))
        ttk.Combobox(sidebar, textvariable=self.output_format, values=["png", "jpg", "webp"], state="readonly").pack(fill=X)

        Button(sidebar, text="Choose Output Folder", command=self.select_output_folder).pack(fill=X, pady=5)
        Button(sidebar, text="Pick Background Color", command=self.pick_bg_color).pack(fill=X, pady=5)

        Label(sidebar, text="Quality (JPG/WebP):", bg="#3a3a3a", fg="white").pack(anchor=W, pady=(10,0))
        Scale(sidebar, from_=1, to=100, orient=HORIZONTAL, variable=self.quality).pack(fill=X)

        Button(sidebar, text="Start Processing", bg="#4CAF50", fg="white", command=self.start_processing).pack(fill=X, pady=10)

        # Progress indicator
        self.progress_frame = Frame(sidebar, bg="#3a3a3a")
        self.progress_frame.pack(fill=X, pady=(10,0))
        self.progress_label = Label(self.progress_frame, text="Idle", bg="#3a3a3a", fg="#cccccc", font=("Arial", 10))
        self.progress_label.pack(anchor="w", pady=(0,4))
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=X)

        # Scrollable work area (dark background)
        container = Frame(self.root, bg="#2e2e2e")
        container.pack(side=RIGHT, expand=True, fill=BOTH)

        self.canvas = Canvas(container, bg="#2e2e2e", highlightthickness=0)
        self.vscroll = Scrollbar(container, orient=VERTICAL, command=self.canvas.yview)
        self.vscroll.pack(side=RIGHT, fill=Y)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.work_area = Frame(self.canvas, bg="#2e2e2e")
        self.canvas.create_window((0, 0), window=self.work_area, anchor="nw")
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Sections: Originals and Processed
        self.before_title = Label(self.work_area, text="Original Images", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
        self.before_title.pack(anchor='w', padx=16, pady=(16, 8))
        self.before_frame = Frame(self.work_area, bg="#2e2e2e")
        self.before_frame.pack(fill=X, padx=12)

        self.divider = Frame(self.work_area, height=2, bg="#aaaaaa")
        self.divider.pack(fill=X, padx=8, pady=16)

        self.after_title = Label(self.work_area, text="Processed Images", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
        self.after_title.pack(anchor='w', padx=16, pady=(0, 8))
        self.after_frame = Frame(self.work_area, bg="#2e2e2e")
        self.after_frame.pack(fill=X, padx=12, pady=(0, 16))


        #Load icons 
        try:
            close_icon = Image.open(r"C:/users/evere/Pictures/x.png")
            close_icon = close_icon.resize((20, 20), Image.LANCZOS)  # Resize for cleaner scaling
            self.remove_icon = ImageTk.PhotoImage(close_icon)
        except Exception as e:
            self.remove_icon = None
            print("Failed to load remove icon:", e)

        try:
            rotate_icon = Image.open(r"C:/users/evere/Pictures/r.png")
            rotate_icon = rotate_icon.resize((20, 20), Image.LANCZOS)  # Resize for cleaner scaling
            self.rotate_icon = ImageTk.PhotoImage(rotate_icon)
        except Exception as e:
            self.rotate_icon = None
            print("Failed to load rotate icon:", e)

    
    # ------------- File pickers -------------
    def select_images(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp")])
        if not paths:
            return
        # extend list and init rotations
        for p in paths:
            if p not in self.images:
                self.images.append(p)
            if p not in self.image_rotations:
                self.image_rotations[p] = 0
        # Append only new items to preview
        new_paths = [p for p in paths if p not in self.before_shown]
        if new_paths:
            self.add_thumbnails(self.before_frame, new_paths, section='before')
        self._refresh_scrollregion()

    
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)

    def pick_bg_color(self):
        c = colorchooser.askcolor(title="Choose Background Color")
        if c[1]:
            self.bg_color = c[1]

    # ------------- Thumbnails -------------
    def _grid_pos(self, index, cols=4):
        return index // cols, index % cols

    def remove_image(self, img_path, tile):
        # Remove image data from tracking structures
        if img_path in self.images:
            self.images.remove(img_path)
        if img_path in self.before_shown:
            self.before_shown.remove(img_path)
        if img_path in self.image_rotations:
            del self.image_rotations[img_path]

        # Destroy the tile frame
        tile.destroy()

        # Refresh the scroll region after removal
        self._refresh_scrollregion()


    def add_thumbnails(self, parent, paths, section='before', thumb_size=150):
        cols = 4
        # determine start index based on what is already shown in that section
        if section == 'before':
            base_index = len(self.before_shown)
        else:
            base_index = len(self.after_shown)

        for offset, img_path in enumerate(paths):
            idx = base_index + offset
            r, c = self._grid_pos(idx, cols)

            # Container per tile
            tile = Frame(parent, bg="#2e2e2e")
            tile.grid(row=r, column=c, padx=12, pady=12, sticky='nw')

            # Image preview (respect stored rotation)
            try:
                img = Image.open(img_path)
                angle = self.image_rotations.get(img_path, 0)
                if angle:
                    img = img.rotate(angle, expand=True)
                img.thumbnail((thumb_size, thumb_size))
                photo = ImageTk.PhotoImage(img)
            except Exception:
                continue

            self.photo_refs.append(photo)
            lbl = Label(tile, image=photo, bg="#2e2e2e")
            lbl.image = photo
            lbl.pack()

            # # Rotate button in top-right of tile
            # Button(
            #     tile, text="↻", font=("Arial", 15, "bold"),
            #     command=lambda p=img_path, l=lbl, s=thumb_size: self.rotate_image(p, l, s),
            #     bg="#555555", fg="white", relief=FLAT, padx=4, pady=1
            # ).place(relx=1, rely=0, anchor='ne')

            if section == 'before' and self.rotate_icon:
                Button(
                    tile, image=self.rotate_icon,
                    command=lambda p=img_path, l=lbl, s=thumb_size: self.rotate_image(p, l, s),
                    bg="#2e2e2e", activebackground="#444444",
                    relief=FLAT, bd=0, highlightthickness=0, padx=0, pady=0
                ).place(relx=1, rely=0, anchor='ne')

            if section == 'before' and self.remove_icon:
                Button(
                    tile, image=self.remove_icon,
                    command=lambda p=img_path, t=tile: self.remove_image(p, t),
                    bg="#2e2e2e", activebackground="#444444",
                    relief=FLAT, bd=0, highlightthickness=0, padx=0, pady=0
                ).place(relx=0, rely=0, anchor='nw')
                                
            # Caption
            size_kb = os.path.getsize(img_path) // 1024 if os.path.exists(img_path) else 0
            caption = f"{os.path.basename(img_path)} ({size_kb} KB)"
            Label(tile, text=caption, bg="#2e2e2e", fg="white", font=("Arial", 10, "bold"), wraplength=thumb_size).pack(pady=(4,0))

            # Track shown
            if section == 'before':
                self.before_shown.add(img_path)
            else:
                self.after_shown.add(img_path)

    def rotate_image(self, img_path, label, preview_size):
        # Update rotation state
        self.image_rotations[img_path] = (self.image_rotations.get(img_path, 0) - 90) % 360
        # Refresh preview
        try:
            img = Image.open(img_path)
            img = img.rotate(self.image_rotations[img_path], expand=True)
            img.thumbnail((preview_size, preview_size))
            photo = ImageTk.PhotoImage(img)
            label.configure(image=photo)
            label.image = photo
            self.photo_refs.append(photo)
        finally:
            self._refresh_scrollregion()

    # ------------- Processing -------------
    def start_processing(self):
        if not self.images:
            return
        
        self.clear_processed_images()
        
        t = threading.Thread(target=self.process_images, daemon=True)
        t.start()

    def clear_processed_images(self):
        # Destroy all widgets inside the processed frame
        for widget in self.after_frame.winfo_children():
            widget.destroy()

        # Reset tracking variables if you have any lists or dicts for processed images
        self.after_shown.clear()
        self.image_rotations = {k: v for k, v in self.image_rotations.items() if k in self.before_shown}

        # Update scroll region if inside a canvas
        if hasattr(self, "_refresh_scrollregion"):
            self._refresh_scrollregion()


    def _update_progress_ui(self, idx, total, name):
        self.progress_label.config(text=f"Processing {idx}/{total}: {name}", fg="#ffffff")
        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = idx

    def _set_done_ui(self, count):
        self.progress_label.config(text=f"Processing complete: {count} images saved", fg="#4CAF50")

    def _refresh_scrollregion(self):
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def add_processed_thumbnail(self, path):
        # Append single processed image to the processed section
        self.add_thumbnails(self.after_frame, [path], section='after')
        self._refresh_scrollregion()

    def process_images(self):
        self.processed_images.clear()
        total = len(self.images)
        for idx, img_path in enumerate(self.images, start=1):
            # UI progress update on main thread
            self.root.after(0, self._update_progress_ui, idx-1, total, os.path.basename(img_path))

            # Read original and remove BG
            with open(img_path, "rb") as f:
                raw = f.read()
            output = remove(raw)
            img = Image.open(io.BytesIO(output)).convert("RGBA")

            # Apply rotation after removal
            angle = self.image_rotations.get(img_path, 0)
            if angle:
                img = img.rotate(angle, expand=True)

            # Format handling
            fmt = self.output_format.get()
            if fmt == "jpg":
                bg = Image.new("RGB", img.size, self.bg_color)
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif fmt == "webp":
                img = img.convert("RGBA")

            # Save
            filename = os.path.splitext(os.path.basename(img_path))[0] + f".{fmt}"
            save_path = os.path.join(self.output_folder.get() or os.path.dirname(img_path), filename)
            img.save(save_path, quality=self.quality.get())
            self.processed_images.append(save_path)

            # Update progress to idx
            self.root.after(0, self._update_progress_ui, idx, total, os.path.basename(img_path))
            # Real-time processed preview on main thread
            self.root.after(0, self.add_processed_thumbnail, save_path)

        # Done UI update
        self.root.after(0, self._set_done_ui, len(self.processed_images))

if __name__ == "__main__":
    root = Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()
