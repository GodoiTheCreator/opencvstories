import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import os
import main
from PIL import Image, ImageTk

class OpenCVStoriesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Stories")
        self.root.geometry("800x600")

        self.current_file = None
        self.current_image = None  # For preview in GUI
        self.channel_mode = 'all'

        self.create_widgets()

    def create_widgets(self):
        # Top Frame for Controls
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # File Operations
        tk.Label(control_frame, text="File Operations:").pack(side=tk.LEFT)
        tk.Button(control_frame, text="Load Image/Video", command=self.load_file).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Record Video", command=self.record_video).pack(side=tk.LEFT, padx=5)
        
        # Channel Selection
        tk.Label(control_frame, text=" | Channel:").pack(side=tk.LEFT)
        self.channel_var = tk.StringVar(value='all')
        channels = [('All', 'all'), ('Gray', 'gray'), ('Red', 'r'), ('Green', 'g'), ('Blue', 'b')]
        for text, mode in channels:
            tk.Radiobutton(control_frame, text=text, variable=self.channel_var, value=mode, command=self.update_channel).pack(side=tk.LEFT)

        # Action Buttons Frame
        action_frame = tk.Frame(self.root, padx=10, pady=5)
        action_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(action_frame, text="Blur", command=self.apply_blur).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Sharpen", command=self.apply_sharpen).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Channel Swap", command=self.apply_swap).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Add Sticker", command=self.add_sticker).pack(side=tk.LEFT, padx=5)
        
        # Arithmetic Operations
        tk.Button(action_frame, text="Add Image", command=self.add_image_op).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Subtract Image", command=self.subtract_image_op).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Blend Image", command=self.blend_image_op).pack(side=tk.LEFT, padx=5)

        # Main Display Area
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message):
        self.status_var.set(message)

    def update_channel(self):
        self.channel_mode = self.channel_var.get()
        self.update_status(f"Channel mode set to: {self.channel_mode}")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.current_file = file_path
            self.update_status(f"Loaded: {os.path.basename(file_path)}")
            self.preview_file()

    def preview_file(self):
        if not self.current_file:
            return
        
        file_type = main.detect_file_type(self.current_file)
        if file_type == "image":
            img = cv2.imread(self.current_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(img)
        elif file_type == "video":
            # For video, just show the first frame
            cap = cv2.VideoCapture(self.current_file)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(frame)
            cap.release()

    def display_image(self, img):
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = img.shape[:2]
            scale = min(canvas_width/w, canvas_height/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img = cv2.resize(img, (new_w, new_h))

        self.current_image = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.current_image, anchor=tk.CENTER)

    def show_image_window(self, img, title="Result"):
        """Displays an image in a new Tkinter window with Save/Close options."""
        top = tk.Toplevel(self.root)
        top.title(title)
        
        # Convert BGR to RGB for display
        if len(img.shape) == 3 and img.shape[2] == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_img = img

        # Resize if too big
        h, w = display_img.shape[:2]
        max_h, max_w = 800, 1000
        if h > max_h or w > max_w:
            scale = min(max_h/h, max_w/w)
            new_w, new_h = int(w*scale), int(h*scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        label = tk.Label(top, image=photo)
        label.image = photo # Keep reference
        label.pack(padx=10, pady=10)
        
        btn_frame = tk.Frame(top)
        btn_frame.pack(pady=10)
        
        def save_action():
            if messagebox.askyesno("Save", "Save changes to original file?", parent=top):
                main.save_edited_image(img, self.current_file)
                self.preview_file()
                top.destroy()
        
        tk.Button(btn_frame, text="Save", command=save_action).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Close", command=top.destroy).pack(side=tk.LEFT, padx=5)

    def record_video(self):
        self.update_status("Recording video... Press 'q' in the recording window to stop.")
        output_path = main.record_video(interactive=False)
        if output_path and os.path.exists(output_path):
            if messagebox.askyesno("Save Video", "Video recorded. Do you want to save it?"):
                save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
                if save_path:
                    os.rename(output_path, save_path)
                    self.update_status(f"Video saved to {save_path}")
                    self.current_file = save_path
                    self.preview_file()
                else:
                    os.remove(output_path)
                    self.update_status("Video discarded.")
            else:
                os.remove(output_path)
                self.update_status("Video discarded.")

    def apply_blur(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "No file loaded.")
            return
        
        # We need to call main.apply_filter_blur but it has cv2.imshow/waitKey logic.
        # Ideally we refactor main.py to separate logic.
        # But for now, let's just use the existing function and let it open a CV2 window.
        # However, main.apply_filter_blur calls ask_save_and_overwrite which we made interactive.
        # We want to handle saving in GUI.
        
        # Better approach: Re-implement the logic here using helper functions or call main functions with interactive=False
        # But main.apply_filter_blur doesn't return the result if interactive=False, it just passes.
        
        # So I should implement the logic here for better control.
        file_type = main.detect_file_type(self.current_file)
        if file_type == "image":
            img = cv2.imread(self.current_file)
            if self.channel_mode in ['r', 'g', 'b', 'gray']:
                img = main.show_channel(img, self.channel_mode)
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            
            self.show_image_window(blurred, "Blurred Image")
        elif file_type == "video":
            # For video, just call the main function but we need to handle the save part.
            # Since main.apply_filter_blur is still coupled with saving logic...
            # I'll just call it and let it use the console for now? No, that's bad for GUI.
            # I'll modify main.apply_filter_blur to accept interactive=False and return frames?
            # Or just re-implement the loop here. Re-implementing is safer.
            
            cap = cv2.VideoCapture(self.current_file)
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.channel_mode in ['r', 'g', 'b', 'gray']:
                    frame = main.show_channel(frame, self.channel_mode)
                blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
                frames.append(blurred_frame)
                cv2.imshow("Blurred Video", blurred_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            
            if frames and messagebox.askyesno("Save", "Save changes?"):
                main.save_edited_video(frames, self.current_file, fps)
                self.preview_file()

    def apply_sharpen(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "No file loaded.")
            return
        
        kernel = main.np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        file_type = main.detect_file_type(self.current_file)
        
        if file_type == "image":
            img = cv2.imread(self.current_file)
            if self.channel_mode in ['r', 'g', 'b', 'gray']:
                img = main.show_channel(img, self.channel_mode)
            sharp = cv2.filter2D(img, -1, kernel)
            
            self.show_image_window(sharp, "Sharpened Image")
        elif file_type == "video":
            cap = cv2.VideoCapture(self.current_file)
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.channel_mode in ['r', 'g', 'b', 'gray']:
                    frame = main.show_channel(frame, self.channel_mode)
                sharp_frame = cv2.filter2D(frame, -1, kernel)
                frames.append(sharp_frame)
                cv2.imshow("Sharpened Video", sharp_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            
            if frames and messagebox.askyesno("Save", "Save changes?"):
                main.save_edited_video(frames, self.current_file, fps)
                self.preview_file()

    def apply_swap(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "No file loaded.")
            return
            
        file_type = main.detect_file_type(self.current_file)
        if file_type == "image":
            img = cv2.imread(self.current_file)
            if self.channel_mode == 'all':
                # Swap R and B
                img[:, :, [0, 2]] = img[:, :, [2, 0]]
            elif self.channel_mode in ['r', 'g', 'b', 'gray']:
                img = main.show_channel(img, self.channel_mode)
                
            self.show_image_window(img, "Channel Swap")
        # Video implementation omitted for brevity, similar to above

    def add_sticker(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "No file loaded.")
            return
        
        if main.detect_file_type(self.current_file) != "image":
            messagebox.showwarning("Warning", "Stickers only supported for images.")
            return

        stickers_dir = "stickers"
        if not os.path.exists(stickers_dir):
            messagebox.showerror("Error", "Stickers directory not found.")
            return
            
        stickers = [f for f in os.listdir(stickers_dir) if f.lower().endswith('.png')]
        if not stickers:
            messagebox.showwarning("Warning", "No stickers found.")
            return
            
        # Sticker selection dialog
        sticker_name = simpledialog.askstring("Select Sticker", f"Available stickers:\n{', '.join(stickers)}\nEnter filename:")
        if not sticker_name or sticker_name not in stickers:
            return
            
        sticker_path = os.path.join(stickers_dir, sticker_name)
        sticker_img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
        
        if sticker_img is None:
            return

        # Interactive placement
        img = cv2.imread(self.current_file, cv2.IMREAD_UNCHANGED)
        # Ensure working with BGRA for consistency with overlay function
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
        preview_img = img.copy()
        
        # Create Toplevel window for interaction
        top = tk.Toplevel(self.root)
        top.title("Add Sticker - Click to Place")
        
        # We need to handle resizing for display but map coordinates back to original image
        # For simplicity, let's display at 1:1 if possible, or scrollable?
        # Or just resize for display and map clicks.
        
        # Let's use the show_image_window logic but adapted for interaction
        
        # Convert to RGB for display
        display_img_cv = cv2.cvtColor(preview_img, cv2.COLOR_BGRA2RGBA)
        
        # Resize logic
        h, w = display_img_cv.shape[:2]
        max_h, max_w = 800, 1000
        scale = 1.0
        if h > max_h or w > max_w:
            scale = min(max_h/h, max_w/w)
            new_w, new_h = int(w*scale), int(h*scale)
            display_img_cv = cv2.resize(display_img_cv, (new_w, new_h))
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(display_img_cv))
        
        canvas = tk.Canvas(top, width=display_img_cv.shape[1], height=display_img_cv.shape[0])
        canvas.pack(padx=10, pady=10)
        canvas_image_id = canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        
        # Keep reference to avoid garbage collection
        canvas.image = photo 
        
        def on_click(event):
            nonlocal preview_img, photo
            # Map click coordinates back to original image size
            real_x = int(event.x / scale)
            real_y = int(event.y / scale)
            
            # Center the sticker
            h_sticker, w_sticker = sticker_img.shape[:2]
            real_x -= w_sticker // 2
            real_y -= h_sticker // 2
            
            # Apply sticker
            updated = main.overlay_sticker(preview_img, sticker_img, (real_x, real_y))
            preview_img[:] = updated
            
            # Update display
            display_updated = cv2.cvtColor(preview_img, cv2.COLOR_BGRA2RGBA)
            if scale != 1.0:
                display_updated = cv2.resize(display_updated, (display_img_cv.shape[1], display_img_cv.shape[0]))
            
            new_photo = ImageTk.PhotoImage(image=Image.fromarray(display_updated))
            canvas.itemconfig(canvas_image_id, image=new_photo)
            canvas.image = new_photo # Keep reference
            
        canvas.bind("<Button-1>", on_click)
        
        btn_frame = tk.Frame(top)
        btn_frame.pack(pady=10)
        
        def save_action():
            if messagebox.askyesno("Save", "Save changes to original file?", parent=top):
                final_img = cv2.cvtColor(preview_img, cv2.COLOR_BGRA2BGR)
                main.save_edited_image(final_img, self.current_file)
                self.preview_file()
                top.destroy()
        
        tk.Button(btn_frame, text="Save", command=save_action).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Close", command=top.destroy).pack(side=tk.LEFT, padx=5)
        
        self.update_status("Click on the image to place the sticker.")

    def add_image_op(self):
        self.arithmetic_op('add')

    def subtract_image_op(self):
        self.arithmetic_op('subtract')

    def blend_image_op(self):
        self.arithmetic_op('blend')

    def arithmetic_op(self, op):
        if not self.current_file:
            messagebox.showwarning("Warning", "Load the first image first.")
            return
            
        file2 = filedialog.askopenfilename(title="Select second image")
        if not file2:
            return
            
        img1 = cv2.imread(self.current_file)
        img2 = cv2.imread(file2)
        
        if img1 is None or img2 is None:
            return
            
        # Resize logic from main.py
        min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
        img1 = cv2.resize(img1, (min_shape[1], min_shape[0]))
        img2 = cv2.resize(img2, (min_shape[1], min_shape[0]))
        
        if self.channel_mode in ['r', 'g', 'b', 'gray']:
            img1 = main.show_channel(img1, self.channel_mode)
            img2 = main.show_channel(img2, self.channel_mode)
            
        if op == 'add':
            result = cv2.add(img1, img2)
        elif op == 'subtract':
            result = cv2.subtract(img1, img2)
        elif op == 'blend':
            alpha = simpledialog.askfloat("Input", "Enter alpha (0.0 - 1.0):", minvalue=0.0, maxvalue=1.0, initialvalue=0.5)
            if alpha is None: return
            beta = 1.0 - alpha
            result = cv2.addWeighted(img1, alpha, img2, beta, 0)
            
        self.show_image_window(result, "Arithmetic Result")

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenCVStoriesApp(root)
    root.mainloop()
