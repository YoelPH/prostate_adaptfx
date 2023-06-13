# -*- coding: utf-8 -*-
"""
GUI for 3D adaptive fractionation with minimum and maximum dose
"""

import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

import numpy as np
import pandas as pd
from scipy.stats import gamma
# from adaptive_fractionation_overlap import *


class VerticalScrolledFrame(tk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(
            self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set, height=1000
        )
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)
        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=tk.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind("<Configure>", _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind("<Configure>", _configure_canvas)


class Task(threading.Thread):
    def __init__(self, master, task):
        threading.Thread.__init__(self, target=task)

        if (
            not hasattr(master, "thread_compute")
            or not master.thread_compute.is_alive()
        ):
            master.thread_compute = self
            self.start()


class GUIextended3D:
    def __init__(self, master):
        self.master = master
        master.title("Overlap Adaptive Fractionation")
        self.frame = VerticalScrolledFrame(master)
        self.frame.pack()
        self.frm_probdis = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        self.frm_probdis.pack()
        self.data = []
        self.info_funcs = [self.info1, self.info4, self.info5]
        self.info_buttons = ["btn_path", "btn_mean", "btn_std", "btn_shae", "btn_scale"]
        for idx in range(len(self.info_funcs)):
            globals()[self.info_buttons[idx]] = tk.Button(
                master=self.frm_probdis, text="?", command=self.info_funcs[idx]
            )
            globals()[self.info_buttons[idx]].grid(row=idx + 1, column=4)
        for idx in range(len(self.info_funcs)):
            globals()[self.info_buttons[idx]] = tk.Button(
                master=self.frm_probdis, text="?", command=self.info_funcs[idx]
            )
            globals()[self.info_buttons[idx]].grid(row=idx + 1, column=4)

        self.var_radio = tk.IntVar()
        self.var_radio.set(1)
        self.hyper_insert = tk.Radiobutton(
            master=self.frm_probdis,
            text="hyperparameters",
            justify="left",
            variable=self.var_radio,
            value=1,
            command=self.checkbox1,
        )
        self.hyper_insert.grid(row=0, column=0)
        self.file_insert = tk.Radiobutton(
            master=self.frm_probdis,
            text="prior data",
            justify="left",
            variable=self.var_radio,
            value=2,
            command=self.checkbox1,
        )
        self.file_insert.grid(row=0, column=1)
        

        # open button
        self.lbl_open = tk.Label(
            master=self.frm_probdis, text="load patient data for prior"
        )
        self.lbl_open.grid(row=1, column=0)
        self.btn_open = tk.Button(
            self.frm_probdis, text="Open a File", command=self.select_file
        )
        self.ent_file = tk.Entry(master=self.frm_probdis, width=20)
        self.btn_open.grid(row=1, column=1)


        self.lbl_alpha = tk.Label(
            master=self.frm_probdis, text="shape of gamma distribution (alpha):"
        )
        self.lbl_alpha.grid(row=2, column=0)
        self.ent_alpha = tk.Entry(master=self.frm_probdis, width=30)
        self.ent_alpha.grid(row=2, column=1, columnspan=2)

        self.lbl_beta = tk.Label(
            master=self.frm_probdis, text="scale of gamma distribution (beta):"
        )
        self.lbl_beta.grid(row=3, column=0)
        self.ent_beta = tk.Entry(master=self.frm_probdis, width=30)
        self.ent_beta.grid(row=3, column=1, columnspan=2)

        self.btn_open.configure(state="disabled")
        self.ent_alpha.configure(state="normal")
        self.ent_beta.configure(state="normal")
        self.ent_file.configure(state="disabled")
        self.ent_alpha.insert(0, "2.468531215126044")
        self.ent_beta.insert(0, "0.02584178910588476")

        # produce master with extra option like number of fractions.
        self.frm_extras = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        self.frm_extras.pack()

        self.lbl_fractions = tk.Label(
            master=self.frm_extras, text="Total number of fractions"
        )
        self.lbl_fractions.grid(row=0, column=0)
        self.ent_fractions = tk.Entry(master=self.frm_extras, width=30)
        self.ent_fractions.grid(row=0, column=1, columnspan=2)
        self.ent_fractions.insert(0, "5")
        self.btn_infofrac = tk.Button(
            master=self.frm_extras, text="?", command=self.infofrac
        )
        self.btn_infofrac.grid(row=0, column=3)

        self.lbl_mindose = tk.Label(master=self.frm_extras, text="minimum dose")
        self.lbl_mindose.grid(row=1, column=0)
        self.ent_mindose = tk.Entry(master=self.frm_extras, width=30)
        self.ent_mindose.grid(row=1, column=1, columnspan=2)
        self.ent_mindose.insert(0, "7.25")
        self.btn_mindose = tk.Button(
            master=self.frm_extras, text="?", command=self.infomin
        )
        self.btn_mindose.grid(row=1, column=3)

        self.lbl_maxdose = tk.Label(master=self.frm_extras, text="maximum dose")
        self.lbl_maxdose.grid(row=2, column=0)
        self.ent_maxdose = tk.Entry(master=self.frm_extras, width=30)
        self.ent_maxdose.grid(row=2, column=1, columnspan=2)
        self.ent_maxdose.insert(0, "9.25")
        self.btn_maxdose = tk.Button(
            master=self.frm_extras, text="?", command=self.infomax
        )
        self.btn_maxdose.grid(row=2, column=3)

        # Create a new frame `frm_form` to contain the Label
        # and Entry widgets for entering variable values
        self.frm_form = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        # Pack the frame into the master
        self.frm_form.pack()

        self.frm_buttons = tk.Frame()
        self.frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)

        self.frm_output = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )

        # add label and entry for filename
        self.label = tk.Label(master=self.frm_form, text="file path of prior patients")
        self.ent_file = tk.Entry(master=self.frm_form, width=50)
        self.label.grid(row=0, column=0, sticky="e")
        self.ent_file.grid(row=0, column=1)
        self.info_funcs = [
            self.info10,
            self.info14,
            self.info15
        ]
        self.info_buttons = [
            "self.btn_sf",
            "self.btn_abt",
            "self.btn_abn",
            "self.btn_OARlimit",
            "self.btn_tumorlimit",
            "self.btn_tumorBED",
            "self.btn_OARBED",
        ]
        # List of field labels
        self.labels = [
            "overlaps separated by spaces:",
            "alpha-beta ratio of tumor:",
            "alpha-beta ratio of OAR:",
            "OAR limit:",
            "prescribed tumor dose:",
            "accumulated tumor dose:",
            "accumulated OAR dose:",
        ]
        self.ent_overl = tk.Entry(master=self.frm_form, width=50)
        self.lbl_overl = tk.Label(master=self.frm_form, text=self.labels[0])
        self.example_list = [
            "overlaps separated by space",
            10,
            3,
            90,
            72,
            "only needed if we calculate the dose for a single fraction",
            "only needed if we calculate the dose for a single fraction",
        ]
        self.lbl_overl.grid(row=0, column=0, sticky="e")
        self.ent_overl.grid(row=0, column=1)
        self.ent_overl.insert(0, f"{self.example_list[0]}")
        self.btn_sf = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[0]
        )
        self.btn_sf.grid(row=0, column=2)

        

        self.ent_tumorlimit = tk.Entry(master=self.frm_form, width=50)
        self.lbl_tumorlimit = tk.Label(master=self.frm_form, text=self.labels[4])
        self.lbl_tumorlimit.grid(row=4, column=0, sticky="e")
        self.ent_tumorlimit.grid(row=4, column=1)
        self.ent_tumorlimit.insert(0, f"{self.example_list[1]}")
        self.btn_tumorlimit = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[1]
        )
        self.btn_tumorlimit.grid(row=4, column=2)

        self.ent_tumor_acc = tk.Entry(master=self.frm_form, width=50)
        self.lbl_tumor_acc = tk.Label(master=self.frm_form, text=self.labels[2])
        self.lbl_tumor_acc.grid(row=5, column=0, sticky="e")
        self.ent_tumor_acc.grid(row=5, column=1)
        self.ent_tumor_acc.insert(0, f"{self.example_list[5]}")
        self.btn_tumor_acc = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[2]
        )
        self.btn_tumor_acc.grid(row=5, column=2)


        # Create a new frame `frm_buttons` to contain the compute button
        self.frm_buttons = tk.Frame(master=self.frame.interior)
        self.frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)
        self.btn_compute = tk.Button(
            master=self.frm_buttons,
            text="compute plan",
            command=lambda: Task(self, self.compute_plan),
        )
        self.btn_compute.pack(side=tk.BOTTOM, ipadx=10)
        self.var = tk.IntVar()
        
        self.lbl_info = tk.Label(
            master=self.frm_output,
            text="There are several default values set. Only the sparing factors have to been inserted.\nThis program might take some minutes to calculate",
        )
        self.lbl_info.pack()

        self.frm_output.pack(fill=tk.BOTH, ipadx=10, ipady=10)

        # progressbar
        self.pb = ttk.Progressbar(
            master=self.frm_output, orient="horizontal", mode="determinate", length=500
        )
        # place the progressbar
        self.pb.pack(pady=10)

    def select_file(self):
        filetypes = (("csv files", "*.csv"), ("All files", "*.*"))

        filename = fd.askopenfilename(
            title="Open a file", initialdir="/", filetypes=filetypes
        )

        showinfo(title="Selected File", message=self.filename)

        self.ent_file.insert(0, filename)
        self.data = np.array(pd.read_csv(self.ent_file.get(), sep=";"))
        self.stds = self.data.std(axis=1)
        self.alpha, self.loc, self.beta = gamma.fit(self.stds, floc=0)
        self.ent_alpha.configure(state="normal")
        self.ent_beta.configure(state="normal")
        self.ent_alpha.delete(0, "end")
        self.ent_beta.delete(0, "end")
        self.ent_alpha.insert(0, self.alpha)
        self.ent_beta.insert(0, self.beta)
        self.ent_alpha.configure(state="disabled")
        self.ent_beta.configure(state="disabled")

    def checkbox1(self):
        if self.var_radio.get() == 1:
            self.btn_open.configure(state="disabled")
            self.ent_alpha.configure(state="normal")
            self.ent_beta.configure(state="normal")
            self.ent_file.configure(state="disabled")
        elif self.var_radio.get() == 2:
            self.ent_file.configure(state="normal")
            self.btn_open.configure(state="normal")
            self.ent_alpha.configure(state="disabled")
            self.ent_beta.configure(state="disabled")

    # assign infobutton commands
    def info1(self):
        self.lbl_info[
            "text"
        ] = "Insert the path of your prior patient data in here. \nThis is only needed, if the checkbox for prior data is marked. \nIf not, one can directly insert the hyperparameters below. \nThe file with the prior data must be of the shape n x k,\nwhere each new patient n is on a row and each fraction for patient n is in column k"


    def info4(self):
        self.lbl_info[
            "text"
        ] = "Insert the shape parameter for the gamma distribution."

    def info5(self):
        self.lbl_info[
            "text"
        ] = "Insert the scale parameter for the gamma distribution."

    def infofrac(self):
        self.lbl_info[
            "text"
        ] = "Insert the number of fractions to be delivered to the patient. \n5 fractions is set a standard SBRT treatment."

    def infomin(self):
        self.lbl_info[
            "text"
        ] = "Insert the minimal physical dose that shall be delivered to the PTV95 in one fraction.\nIt is recommended to not put too high minimum dose constraints to allow adaptation"

    def infomax(self):
        self.lbl_info[
            "text"
        ] = "Insert the maximum physical dose that shall be delivered to the PTV95 in one fraction."

    # assign infobutton commands

    def info10(self):
        self.lbl_info[
            "text"
        ] = "Insert the sparing factors that were observed so far.\n The sparing factor of the planning session must be included!.\nThe sparing factors must be separated by spaces e.g.:\n1.1 0.95 0.88\nFor a whole plan 6 sparing factors are needed."

    def info14(self):
        self.lbl_info[
            "text"
        ] = "Insert the prescribed dose to be delivered to the tumor."

    def info15(self):
        self.lbl_info[
            "text"
        ] = "Insert the accumulated tumor dose so far"

    def compute_plan(self):
        self.btn_compute.configure(state="disabled")
        number_of_fractions = int(self.ent_fractions.get())
        alpha = float(self.ent_alpha.get())
        beta = float(self.ent_beta.get())
        min_dose = float(self.ent_mindose.get())
        max_dose = float(self.ent_maxdose.get())
        try:
            global lbl_output
            self.lbl_output.destroy()
        except:
            pass
        try:
            overlaps_str = (self.ent_overl.get()).split()
            overlaps = [float(i) for i in sparing_foverlaps_stractors_str]
            tumor_limit = float(self.ent_tumorlimit.get())
            BED_tumor = float(self.ent_tumor_acc.get())
            [
                optimal_dose,
                total_dose_delivered_tumor,
                total_dose_delivered_OAR,
                tumor_dose,
                OAR_dose,
            ] = intp3.value_eval(
                len(sparing_factors) - 1,
                number_of_fractions,
                BED_OAR,
                BED_tumor,
                sparing_factors,
                abt,
                abn,
                OAR_limit,
                tumor_limit,
                alpha,
                beta,
                min_dose,
                max_dose,
                fixed_prob,
                fixed_mean,
                fixed_std,
            )
            self.lbl_info[
                "text"
            ] = f"The optimal dose for fraction {len(sparing_factors)-1}  = {optimal_dose}\naccumulated dose in tumor = {total_dose_delivered_tumor}\naccumulated dose OAR = {total_dose_delivered_OAR}"
        except ValueError:
            self.lbl_info[
                "text"
            ] = "please enter correct values. Use the ? boxes for further information."
        self.btn_compute.configure(state="normal")

    def checkbox(self):
        if self.var.get() == 0:
            self.ent_BED_tumor.configure(state="disabled")
            self.ent_BED_OAR.configure(state="disabled")
        else:
            self.ent_BED_tumor.configure(state="normal")
            self.ent_BED_OAR.configure(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    GUI = GUIextended3D(root)
    # Start the application
    root.mainloop()