import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
import os
import subprocess

MAX_ROWS = 50000

class CSVExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Material Explorer (Universal)")
        self.root.geometry("1400x750")

        # Sort states
        self.bg_ascending = True
        self.en_ascending = True

        # Data
        self.df = None
        self.filtered_df = None
        self.cif_dir = None
        
        # --- HARDCODED VESTA PATH ---
        self.vesta_path = r"C:\Users\REHNA\Downloads\VESTA-win64\VESTA-win64\VESTA.exe"
        
        self.current_group_col = "None"

        # Layout
        self.build_ui()
        
        # Auto-start file selection
        self.root.after(100, self.initial_load)

    def initial_load(self):
        """Helper to run ask_files on startup."""
        self.ask_files(exit_on_cancel=True)

    # ================= FILE SELECTION =================
    def ask_files(self, exit_on_cancel=False):
        # 1. Ask for CSV
        csv_path = filedialog.askopenfilename(
            title="Step 1: Select your Harvest CSV file",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not csv_path:
            if exit_on_cancel:
                self.root.destroy()
            return

        # 2. Ask for CIF Folder
        initial_dir = os.path.dirname(csv_path)
        cif_dir = filedialog.askdirectory(
            title="Step 2: Select the folder containing CIF files",
            initialdir=initial_dir 
        )
        if not cif_dir:
            if exit_on_cancel:
                self.root.destroy()
            return

        self.cif_dir = cif_dir
        
        try:
            # Reset Data
            self.df = pd.read_csv(csv_path)
            self.normalize_columns()
            self.filtered_df = self.df.copy()
            
            # --- DYNAMIC GROUP UPDATE ---
            # Update the Group By combobox with all column names
            all_cols = ["None"] + list(self.df.columns)
            self.group_combo['values'] = all_cols
            
            # Reset UI Filters
            self.reset_filters_ui()
            
            # Load Data
            self.load_table(self.filtered_df)
            self.update_status()
            
        except Exception as e:
            messagebox.showerror("Error Loading CSV", str(e))
            if exit_on_cancel:
                self.root.destroy()

    def normalize_columns(self):
        """Standardizes column names."""
        self.df.columns = [c.strip() for c in self.df.columns]
        col_map = {
            'file_name': 'File', 'filename': 'File', 'file': 'File',
            'formula': 'Formula',
            'energy': 'Energy', 'final_e': 'Energy',
            'bandgap': 'Band_Gap', 'band_gap': 'Band_Gap', 'gap': 'Band_Gap',
            'type': 'Type', 'mat_type': 'Type',
            'space_group_symbol': 'Symbol',
            'space_group_number': 'Space Group', 'space_group': 'Space Group',
            'application_class': 'Application Class', 'class': 'Application Class',
            'num_atoms': 'NumAtoms'
        }
        
        new_cols = {}
        for col in self.df.columns:
            lower_col = col.lower()
            if lower_col in col_map:
                new_cols[col] = col_map[lower_col]
        
        self.df.rename(columns=new_cols, inplace=True)
        
        # Ensure critical columns exist
        required = ["Type", "Energy", "Band_Gap", "Formula", "File", "Space Group"]
        for req in required:
            if req not in self.df.columns:
                self.df[req] = "N/A"
                if req in ["Energy", "Band_Gap"]:
                    self.df[req] = 0.0

    # ================= UI BUILDER =================
    def build_ui(self):
        # 1. Top Control Bar (File & Grouping)
        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(top_bar, text="📂 OPEN CSV", command=lambda: self.ask_files(False)).pack(side="left", padx=5)
        
        ttk.Label(top_bar, text=" |  Group By:").pack(side="left", padx=5)
        self.group_var = tk.StringVar(value="None")
        
        # Initial empty list, populated on load
        self.group_combo = ttk.Combobox(top_bar, textvariable=self.group_var, values=["None"], state="readonly", width=15)
        self.group_combo.pack(side="left", padx=5)
        self.group_combo.bind("<<ComboboxSelected>>", self.on_group_change)

        # 2. Filter Bar
        filter_frame = ttk.Labelframe(self.root, text="Filters")
        filter_frame.pack(fill="x", padx=5, pady=2)

        # Row 0: Search & Buttons
        f_row1 = ttk.Frame(filter_frame)
        f_row1.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(f_row1, text="Formula:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(f_row1, textvariable=self.search_var, width=15).pack(side="left", padx=5)
        
        ttk.Label(f_row1, text="Type:").pack(side="left", padx=5)
        self.type_var = tk.StringVar(value="All")
        ttk.Combobox(f_row1, textvariable=self.type_var, values=["All", "Metal", "Semiconductor", "Insulator"], width=12, state="readonly").pack(side="left")

        ttk.Button(f_row1, text="Apply Filters", command=self.apply_filters).pack(side="left", padx=15)
        ttk.Button(f_row1, text="Reset", command=self.reset_filters).pack(side="left")

        # Row 1: Ranges
        f_row2 = ttk.Frame(filter_frame)
        f_row2.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(f_row2, text="Band Gap:").pack(side="left")
        self.bg_min = tk.StringVar()
        self.bg_max = tk.StringVar()
        ttk.Entry(f_row2, textvariable=self.bg_min, width=6).pack(side="left", padx=2)
        ttk.Label(f_row2, text="-").pack(side="left")
        ttk.Entry(f_row2, textvariable=self.bg_max, width=6).pack(side="left", padx=2)

        ttk.Label(f_row2, text="Energy:").pack(side="left", padx=10)
        self.en_min = tk.StringVar()
        self.en_max = tk.StringVar()
        ttk.Entry(f_row2, textvariable=self.en_min, width=6).pack(side="left", padx=2)
        ttk.Label(f_row2, text="-").pack(side="left")
        ttk.Entry(f_row2, textvariable=self.en_max, width=6).pack(side="left", padx=2)

        ttk.Button(f_row2, text="Sort Gap", command=self.sort_by_band_gap).pack(side="left", padx=10)
        ttk.Button(f_row2, text="Sort Energy", command=self.sort_by_energy).pack(side="left")
        
        # 3. Table Area
        table_frame = ttk.Frame(self.root)
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)

        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")

        self.tree = ttk.Treeview(table_frame, show="headings", yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Bindings
        self.tree.bind("<Double-1>", lambda e: self.view_structure())
        self.tree.bind("<Button-3>", self.show_context_menu)  # Right click menu

        # Context Menu
        self.context_menu = Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy Row", command=self.copy_selected_row)

        # Status Bar
        self.status = ttk.Label(self.root, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=5, pady=2)

    # ================= LOADING & GROUPING =================
    def load_table(self, df):
        # Clear Table
        self.tree.delete(*self.tree.get_children())
        
        # Define Columns
        cols = list(df.columns)
        self.tree["columns"] = cols
        
        group_col = self.current_group_col
        
        # 1. GROUPING MODE
        if group_col != "None" and group_col in df.columns:
            # Enable Tree Column for groups
            self.tree["show"] = "tree headings"
            self.tree.heading("#0", text="Category")
            self.tree.column("#0", width=150, anchor="w")
            
            # Setup Headings
            for col in cols:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, anchor="center")

            # Group Data
            grouped = df.groupby(group_col)
            
            for group_name, group_data in grouped:
                # Insert Parent Node
                parent_id = self.tree.insert("", "end", text=f"{group_name} ({len(group_data)})", open=True)
                
                # Insert Children
                for _, row in group_data.head(MAX_ROWS).iterrows():
                    self.tree.insert(parent_id, "end", values=list(row))

        # 2. FLAT MODE (Standard)
        else:
            self.tree["show"] = "headings"
            for col in cols:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, anchor="center")
            
            for _, row in df.head(MAX_ROWS).iterrows():
                self.tree.insert("", "end", values=list(row))
        
        self.update_status()

    def on_group_change(self, event):
        """Triggered when Group By dropdown changes."""
        self.current_group_col = self.group_var.get()
        if self.filtered_df is not None:
            self.load_table(self.filtered_df)

    def reset_filters_ui(self):
        self.search_var.set("")
        self.type_var.set("All")
        self.bg_min.set("")
        self.bg_max.set("")
        self.en_min.set("")
        self.en_max.set("")
        self.group_var.set("None")
        self.current_group_col = "None"

    def update_status(self):
        """Updates the status bar with current row counts."""
        if self.df is None:
            self.status.config(text="Ready")
            return
            
        total = len(self.df)
        visible = len(self.filtered_df) if self.filtered_df is not None else 0
        self.status.config(text=f"Showing: {visible} / {total} Rows")

    # ================= FILTER LOGIC =================
    def apply_filters(self):
        if self.df is None: return
        df = self.df.copy()
        
        # Formula
        query = self.search_var.get().strip()
        if query and "Formula" in df.columns:
            df = df[df["Formula"].astype(str).str.contains(query, case=False, na=False, regex=False)]

        # Type
        if self.type_var.get() != "All" and "Type" in df.columns:
            df = df[df["Type"] == self.type_var.get()]

        # Ranges
        try:
            if "Band_Gap" in df.columns:
                if self.bg_min.get(): df = df[df["Band_Gap"] >= float(self.bg_min.get())]
                if self.bg_max.get(): df = df[df["Band_Gap"] <= float(self.bg_max.get())]
            if "Energy" in df.columns:
                if self.en_min.get(): df = df[df["Energy"] >= float(self.en_min.get())]
                if self.en_max.get(): df = df[df["Energy"] <= float(self.en_max.get())]
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric range")
            return

        self.filtered_df = df
        self.load_table(df)

    def reset_filters(self):
        self.reset_filters_ui()
        self.filtered_df = self.df.copy()
        self.load_table(self.filtered_df)

    # ================= SORTING =================
    def sort_by_band_gap(self):
        if self.filtered_df is None or "Band_Gap" not in self.filtered_df.columns: return
        self.filtered_df = self.filtered_df.sort_values(by="Band_Gap", ascending=self.bg_ascending)
        self.bg_ascending = not self.bg_ascending
        self.load_table(self.filtered_df)

    def sort_by_energy(self):
        if self.filtered_df is None or "Energy" not in self.filtered_df.columns: return
        self.filtered_df = self.filtered_df.sort_values(by="Energy", ascending=self.en_ascending)
        self.en_ascending = not self.en_ascending
        self.load_table(self.filtered_df)

    # ================= VESTA & UTILS =================
    def show_context_menu(self, event):
        """Show right-click menu."""
        row_id = self.tree.identify_row(event.y)
        if row_id:
            self.tree.selection_set(row_id)
            self.context_menu.post(event.x_root, event.y_root)

    def copy_selected_row(self):
        """Copy the selected row values to clipboard."""
        sel = self.tree.focus()
        if sel:
            item = self.tree.item(sel)
            if item["values"]:
                row_str = ", ".join(map(str, item["values"]))
                self.root.clipboard_clear()
                self.root.clipboard_append(row_str)
                self.root.update()

    def view_structure(self):
        sel = self.tree.focus()
        if not sel: return

        item = self.tree.item(sel)
        # Prevent opening "Group" parent nodes
        if not item["values"]: 
            return 

        # Map values back to columns
        # Column mapping relies on tree columns matching DF columns order
        row_vals = item["values"]
        cols = self.tree["columns"]
        
        # Safe mapping
        row = {}
        for i, col in enumerate(cols):
            if i < len(row_vals):
                row[col] = row_vals[i]

        cif_name = row.get("File")
        if not cif_name: return

        cif_path = os.path.join(self.cif_dir, str(cif_name))
        
        # Locate VESTA if needed
        if not os.path.exists(self.vesta_path):
            messagebox.showinfo("Locate VESTA", "VESTA executable not found at default path.\nPlease select VESTA.exe manually.")
            exe = filedialog.askopenfilename(filetypes=[("Executable", "*.exe")])
            if exe: 
                self.vesta_path = exe
            else: 
                return

        if os.path.exists(cif_path):
            try:
                subprocess.Popen([self.vesta_path, cif_path])
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showerror("Error", f"CIF file not found:\n{cif_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVExplorer(root)
    root.mainloop()