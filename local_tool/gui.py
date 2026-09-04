"""
Double-click GUI for the manhwa panel extractor - no terminal needed.
Paste a chapter URL, pick where to save, click Download.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import Config
from runner import run_chapter, Cancelled

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Manhwa Panels")


def open_in_file_manager(path):
    if sys.platform == "darwin":
        subprocess.run(["open", path])
    elif os.name == "nt":
        os.startfile(path)  # noqa: (Windows-only call, guarded by os.name check)
    else:
        subprocess.run(["xdg-open", path])


class App:
    def __init__(self, root):
        self.root = root
        root.title("Manhwa Panel Downloader")
        root.geometry("640x520")
        root.minsize(520, 420)

        self.events = queue.Queue()
        self.cancel_flag = threading.Event()
        self.worker = None
        self.last_output_dir = None

        pad = {"padx": 10, "pady": 6}

        form = ttk.Frame(root)
        form.pack(fill="x", **pad)

        ttk.Label(form, text="Chapter URL").grid(row=0, column=0, sticky="w")
        self.url_entry = ttk.Entry(form)
        self.url_entry.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.url_entry.focus()

        ttk.Label(form, text="Save to").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.output_entry = ttk.Entry(form, textvariable=self.output_var)
        self.output_entry.grid(row=3, column=0, sticky="ew")
        ttk.Button(form, text="Browse...", command=self.browse_output).grid(row=3, column=1, padx=(6, 0))

        form.columnconfigure(0, weight=1)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", **pad)
        self.download_btn = ttk.Button(buttons, text="Download", command=self.start_download)
        self.download_btn.pack(side="left")
        self.cancel_btn = ttk.Button(buttons, text="Cancel", command=self.cancel_download, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.open_folder_btn = ttk.Button(buttons, text="Open Folder", command=self.open_last_output, state="disabled")
        self.open_folder_btn.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(root, mode="determinate")
        self.progress.pack(fill="x", **pad)

        self.log_box = scrolledtext.ScrolledText(root, state="disabled", wrap="word", height=16)
        self.log_box.pack(fill="both", expand=True, **pad)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0, 8))

        root.bind("<Return>", lambda e: self.start_download())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._poll_queue()

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def browse_output(self):
        chosen = filedialog.askdirectory(initialdir=self.output_var.get() or os.path.expanduser("~"))
        if chosen:
            self.output_var.set(chosen)

    def start_download(self):
        if self.worker and self.worker.is_alive():
            return
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Paste a chapter URL first.")
            return
        output_dir = self.output_var.get().strip() or DEFAULT_OUTPUT_DIR
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Can't use that folder", str(exc))
            return

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("Working...")
        self.download_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_folder_btn.configure(state="disabled")
        self.cancel_flag.clear()

        self.worker = threading.Thread(target=self._run, args=(url, output_dir), daemon=True)
        self.worker.start()

    def _run(self, url, output_dir):
        def log(message):
            self.events.put(("log", message))

        def progress(done, total):
            self.events.put(("progress", (done, total)))

        try:
            metadata = run_chapter(
                output_dir=output_dir,
                url=url,
                config=Config(),
                log=log,
                progress=progress,
                should_cancel=self.cancel_flag.is_set,
            )
            self.events.put(("done", metadata))
        except Cancelled:
            self.events.put(("cancelled", None))
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            self.events.put(("error", str(exc)))

    def cancel_download(self):
        self.cancel_flag.set()
        self.status_var.set("Cancelling...")
        self.cancel_btn.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    done, total = payload
                    if total:
                        self.progress.configure(maximum=total, value=done)
                        self.status_var.set(f"Extracting panels... ({done}/{total})")
                elif kind == "done":
                    self.last_output_dir = payload.get("output_dir")
                    self.status_var.set(f"Done - {len(payload.get('panels', []))} panels extracted.")
                    self.progress.configure(value=self.progress["maximum"])
                    self._finish()
                    self.open_folder_btn.configure(state="normal")
                elif kind == "cancelled":
                    self.status_var.set("Cancelled.")
                    self._finish()
                elif kind == "error":
                    self.status_var.set("Something went wrong.")
                    self.log(f"Error: {payload}")
                    messagebox.showerror("Download failed", payload)
                    self._finish()
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _finish(self):
        self.download_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    def open_last_output(self):
        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            open_in_file_manager(self.last_output_dir)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Download in progress", "A download is still running. Quit anyway?"):
                return
            self.cancel_flag.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
