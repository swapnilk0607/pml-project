import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import csv

class ImageTaggerApp:
    def __init__(self, root, image_folder, output_csv="wtc_tagged_data.csv", processed_folder="processed_images"):
        self.root = root
        self.root.title("Image Grid Tagger Tool - 8x8 Grid")
        self.root.geometry("1000x700")
        # Ensure window appears on top and focused when launched
        try:
            self.root.attributes('-topmost', True)
            self.root.update()
            self.root.after(200, lambda: self.root.attributes('-topmost', False))
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

        self.image_folder = image_folder
        self.output_csv = output_csv
        self.processed_folder = processed_folder
        
        # Create processed_images folder if it doesn't exist
        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)
        
        # Grid configuration
        self.grid_size = 8  # 8x8 grid
        self.tagged_cells = {}  # {(row, col): category}
        self.current_category = 1  # Default: Ball
        
        # Category colors and names
        self.categories = {
            1: {"name": "Ball", "color": "#FF0000"},
            2: {"name": "Bat", "color": "#0000FF"},
            3: {"name": "Stump", "color": "#00FF00"}
        }
        
        # Load images
        self.image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.current_index = 0
        
        if not self.image_list:
            messagebox.showerror("Error", "No images found in the specified folder.")
            root.destroy()
            return

        # --- UI Layout ---
        
        # Left Frame (Image with Grid)
        self.left_frame = tk.Frame(root, width=700, height=700, bg="#e1e1e1")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)
        
        # Canvas for interactive grid
        self.canvas = tk.Canvas(self.left_frame, bg="#e1e1e1", highlightthickness=0)
        self.canvas.pack(expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Right Frame (Controls)
        self.right_frame = tk.Frame(root, width=300, bg="white", padx=20, pady=20)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_frame.pack_propagate(False)
        
        # Category Selection
        self.create_category_selector()
        
        # Cell count display
        self.create_cell_counter()
        
        # Control Buttons
        self.create_control_buttons()
        
        # Status
        self.status_label = tk.Label(self.right_frame, text="", bg="white", fg="#666666", font=("Arial", 9))
        self.status_label.pack(side=tk.BOTTOM, pady=10)

        # Initialize CSV if not exists
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Create header with columns for up to 64 cells (8x8 grid)
                header = ["imagename"]
                for i in range(1, 65):  # Support up to 64 tagged cells
                    header.extend([f"cell_{i}_row", f"cell_{i}_column", f"cell_{i}_object_tag"])
                writer.writerow(header)

        self.load_image()

    def create_category_selector(self):
        """Creates category selection UI"""
        tk.Label(self.right_frame, text="Select Category", font=("Arial", 16, "bold"), bg="white", fg="black").pack(pady=(0, 15))
        
        self.category_var = tk.IntVar(value=1)
        
        for cat_id, cat_info in self.categories.items():
            frame = tk.Frame(self.right_frame, bg="white")
            frame.pack(fill=tk.X, pady=5)
            
            rb = tk.Radiobutton(
                frame,
                text=f"{cat_id}. {cat_info['name']}",
                variable=self.category_var,
                value=cat_id,
                bg="white",
                fg="black",
                font=("Arial", 12),
                command=self.update_current_category,
                activebackground="white",
                activeforeground="black",
                selectcolor="white"
            )
            rb.pack(side=tk.LEFT)
            
            # Color indicator
            color_box = tk.Label(frame, bg=cat_info['color'], width=3, height=1, relief=tk.SOLID, borderwidth=1)
            color_box.pack(side=tk.RIGHT, padx=5)
        
        tk.Label(self.right_frame, text="", bg="white").pack(pady=5)  # Spacer

    def create_cell_counter(self):
        """Creates cell count display"""
        tk.Label(self.right_frame, text="Tagged Cells", font=("Arial", 14, "bold"), bg="white", fg="black").pack(pady=(10, 10))
        
        self.counter_frame = tk.Frame(self.right_frame, bg="white")
        self.counter_frame.pack(fill=tk.X, pady=5)
        
        self.counter_labels = {}
        for cat_id, cat_info in self.categories.items():
            frame = tk.Frame(self.counter_frame, bg="white")
            frame.pack(fill=tk.X, pady=3)
            
            tk.Label(frame, text=f"{cat_info['name']}:", bg="white", fg="black", font=("Arial", 10), width=8, anchor="w").pack(side=tk.LEFT)
            
            count_label = tk.Label(frame, text="0", bg="white", font=("Arial", 10, "bold"), fg=cat_info['color'])
            count_label.pack(side=tk.LEFT)
            
            self.counter_labels[cat_id] = count_label
        
        tk.Label(self.right_frame, text="", bg="white").pack(pady=5)  # Spacer

    def create_control_buttons(self):
        """Creates control buttons with forced non-native rendering for dark mode compatibility"""
        self.btn_frame = tk.Frame(self.right_frame, bg="white", pady=20)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Clear Grid Button - Orange with white text
        self.btn_clear = tk.Button(
            self.btn_frame, 
            text="Clear Grid", 
            command=self.clear_grid, 
            bg="#E0E0E0", 
            fg="black", 
            font=("Arial", 11),
            activebackground="#BDBDBD",
            activeforeground="black",
            relief=tk.RAISED,  # Force non-native rendering
            borderwidth=2,
            cursor="hand2"
        )
        self.btn_clear.pack(fill=tk.X, pady=5)
        
        # Save & Next Button - Green with white text
        self.btn_save = tk.Button(
            self.btn_frame, 
            text="Save & Next", 
            command=self.save_and_next, 
           bg="#E0E0E0", 
            fg="black", 
            font=("Arial", 11),
            activebackground="#BDBDBD",
            activeforeground="black",
            relief=tk.RAISED,  # Force non-native rendering
            borderwidth=2,
            cursor="hand2"
            
        )
        self.btn_save.pack(fill=tk.X, pady=5)
        
        # Skip Image Button - Gray with black text
        self.btn_skip = tk.Button(
            self.btn_frame, 
            text="Skip Image", 
            command=self.next_image, 
            bg="#E0E0E0", 
            fg="black", 
            font=("Arial", 11),
            activebackground="#BDBDBD",
            activeforeground="black",
            relief=tk.RAISED,  # Force non-native rendering
            borderwidth=2,
            cursor="hand2"
        )
        self.btn_skip.pack(fill=tk.X, pady=5)

    def update_current_category(self):
        """Updates the currently selected category"""
        self.current_category = self.category_var.get()

    def load_image(self):
        """Loads and displays the current image with grid overlay"""
        if self.current_index >= len(self.image_list):
            messagebox.showinfo("Done", "All images have been processed!")
            self.root.destroy()  # Terminate the program completely
            return

        filename = self.image_list[self.current_index]
        path = os.path.join(self.image_folder, filename)
        
        # Reset tagged cells for new image
        self.tagged_cells = {}
        self.update_cell_counters()
        
        # Update Status
        self.status_label.config(text=f"Image {self.current_index + 1} of {len(self.image_list)}\n{filename}")

        # Open and Resize Image
        try:
            img = Image.open(path)
            
            # Calculate display size (square for grid)
            display_size = 640
            img.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            
            self.current_image = img
            self.tk_image = ImageTk.PhotoImage(img)
            
            # Store image dimensions for grid calculation
            self.img_width = img.width
            self.img_height = img.height
            
            # Clear and redraw canvas
            self.draw_canvas()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def draw_canvas(self):
        """Draws the image and grid on canvas"""
        self.canvas.delete("all")
        
        # Calculate canvas size and position
        canvas_width = self.img_width + 20
        canvas_height = self.img_height + 20
        self.canvas.config(width=canvas_width, height=canvas_height)
        
        # Draw image centered
        self.canvas.create_image(10, 10, image=self.tk_image, anchor=tk.NW)
        
        # Draw grid
        self.draw_grid()

    def draw_grid(self):
        """Draws the 8x8 grid overlay and tagged cell borders"""
        cell_width = self.img_width / self.grid_size
        cell_height = self.img_height / self.grid_size
        
        # Draw grid lines (light gray)
        for i in range(self.grid_size + 1):
            # Vertical lines
            x = 10 + i * cell_width
            self.canvas.create_line(x, 10, x, 10 + self.img_height, fill="#CCCCCC", width=1, tags="grid")
            
            # Horizontal lines
            y = 10 + i * cell_height
            self.canvas.create_line(10, y, 10 + self.img_width, y, fill="#CCCCCC", width=1, tags="grid")
        
        # Draw tagged cell borders
        for (row, col), category in self.tagged_cells.items():
            x1 = 10 + col * cell_width
            y1 = 10 + row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            color = self.categories[category]["color"]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=4, tags="tagged")

    def on_canvas_click(self, event):
        """Handles click events on the canvas"""
        # Calculate which cell was clicked
        x = event.x - 10
        y = event.y - 10
        
        # Check if click is within image bounds
        if x < 0 or y < 0 or x >= self.img_width or y >= self.img_height:
            return
        
        cell_width = self.img_width / self.grid_size
        cell_height = self.img_height / self.grid_size
        
        col = int(x / cell_width)
        row = int(y / cell_height)
        
        # Tag the cell with current category
        self.tagged_cells[(row, col)] = self.current_category
        
        # Redraw grid to show updated borders
        self.draw_grid()
        
        # Update counters
        self.update_cell_counters()

    def update_cell_counters(self):
        """Updates the cell count display"""
        counts = {1: 0, 2: 0, 3: 0}
        for category in self.tagged_cells.values():
            counts[category] += 1
        
        for cat_id, count in counts.items():
            self.counter_labels[cat_id].config(text=str(count))

    def clear_grid(self):
        """Clears all tagged cells"""
        self.tagged_cells = {}
        self.draw_grid()
        self.update_cell_counters()

    def save_tagged_image(self, filename):
        """Saves the current image with grid overlay and tagged cells to processed_images folder"""
        try:
            # Load the original image
            img_path = os.path.join(self.image_folder, filename)
            img = Image.open(img_path)
            
            # Create a copy to draw on
            img_with_grid = img.copy()
            draw = ImageDraw.Draw(img_with_grid)
            
            # Calculate cell dimensions
            img_width, img_height = img.size
            cell_width = img_width / self.grid_size
            cell_height = img_height / self.grid_size
            
            # Draw grid lines (light gray)
            grid_color = "#CCCCCC"
            for i in range(self.grid_size + 1):
                # Vertical lines
                x = i * cell_width
                draw.line([(x, 0), (x, img_height)], fill=grid_color, width=2)
                
                # Horizontal lines
                y = i * cell_height
                draw.line([(0, y), (img_width, y)], fill=grid_color, width=2)
            
            # Draw tagged cell borders (thick colored borders)
            for (row, col), category in self.tagged_cells.items():
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                color = self.categories[category]["color"]
                # Draw thick border (6 pixels)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            
            # Save to processed_images folder
            output_path = os.path.join(self.processed_folder, filename)
            img_with_grid.save(output_path)
            print(f"Saved tagged image to: {output_path}")
            
        except Exception as e:
            print(f"Error saving tagged image: {e}")

    def save_and_next(self):
        """Saves tagged grid data to CSV - one row per image with columns for each cell"""
        filename = self.image_list[self.current_index]
        
        if not self.tagged_cells:
            response = messagebox.askyesno("No Tags", "No cells have been tagged. Save anyway?")
            if not response:
                return
        
        # Prepare row data starting with filename
        row_data = [filename]
        
        # Add each tagged cell as three columns: row, column, object_tag
        for (row, col), category in sorted(self.tagged_cells.items()):
            object_tag = self.categories[category]["name"]
            row_data.extend([row, col, object_tag])
        
        # Pad with empty values for unused cell columns (up to 64 cells)
        num_tagged = len(self.tagged_cells)
        remaining_cells = 64 - num_tagged
        for _ in range(remaining_cells):
            row_data.extend(["", "", ""])  # Empty row, column, tag
        
        # Save to CSV (one row per image)
        try:
            with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            
            print(f"Saved {len(self.tagged_cells)} tagged cells for {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save to CSV:\n{e}")
            return
        
        # Save the tagged image with grid overlay
        self.save_tagged_image(filename)
        
        self.next_image()

    def next_image(self):
        """Moves to the next image"""
        self.current_index += 1
        self.load_image()

def launch_annotation_tool(image_folder, output_csv="wtc_tagged_data.csv", processed_folder="processed_images"):
    """
    Launch the image annotation tool with specified folders.
    
    Args:
        image_folder (str): Path to folder containing images to annotate
        output_csv (str): Path to output CSV file for storing annotations
        processed_folder (str): Path to folder for saving annotated images
    
    Returns:
        None
    """
    # Create folders if they don't exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Created folder '{image_folder}'. Please add images to it.")
    
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Launch the annotation GUI
    root = tk.Tk()
    app = ImageTaggerApp(root, image_folder, output_csv, processed_folder)
    root.mainloop()


def get_annotation_data(csv_path="wtc_tagged_data.csv"):
    """
    Read annotation data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing annotations
    
    Returns:
        list: List of dictionaries containing annotation data
    """
    import csv
    
    if not os.path.exists(csv_path):
        return []
    
    annotations = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(row)
    
    return annotations


if __name__ == "__main__":
    # --- CONFIGURATION ---
    folder_path = "./input"  # Folder containing images to tag
    
    # Launch with default settings
    launch_annotation_tool(
        image_folder=folder_path,
        output_csv="./tagged_data.csv",
        processed_folder="./output"
    )